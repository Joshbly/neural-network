;; The brains' forward pass for js/nn.js. Build with `node train/build-nn.js` (writes js/nn-wasm.js).
;;
;; Bit-identical to Brain.thinkJS: each SIMD lane is one neuron and does exactly what the JS loop does for
;; it, in the same order and precision: a float64 sum that starts at the bias and adds weight x input one
;; input at a time, the same squash written in the same operation order, and float32 between layers. The
;; speed comes from running 8 neurons side by side with independent accumulators (and, in forward2, both
;; mirror passes off one load of each weight), never from reordering a sum.
;;
;; Memory: two scratch buffers at SPLAT (1024) and SPLAT + 16384 hold the current layer's inputs as
;; float64 pairs, so each is converted once per layer instead of once per block. Brains live above 33792:
;; js/nn.js writes each one a descriptor (the layer count, then per layer
;;   [weights, input count, blocks of 8 neurons, actsIn, actsOut, mirrorIn, mirrorOut])
;; and its weights as float64 (exact copies of the float32 genes) in blocks of 8 neurons: the 8 biases,
;; then for each input the 8 weights from it. Layers are padded to whole blocks with all-zero neurons
;; whose outputs nothing reads. Activations are float32, which is what the JS loop rounds them to.
(module
  ;; fixed size, so the typed-array views js/nn.js keeps into it never detach
  (memory (export "memory") 256 256)

  ;; squash(x) = x <= -3 ? -1 : x >= 3 ? 1 : x * (27 + x * x) / (27 + 9 * x * x), rounded to float32
  (func $emit (param $p i32) (param $x v128)
    (local $r v128)
    (local.set $r
      (f64x2.div
        (f64x2.mul (local.get $x)
          (f64x2.add (v128.const f64x2 27 27) (f64x2.mul (local.get $x) (local.get $x))))
        (f64x2.add (v128.const f64x2 27 27)
          (f64x2.mul (f64x2.mul (v128.const f64x2 9 9) (local.get $x)) (local.get $x)))))
    (local.set $r
      (v128.bitselect (v128.const f64x2 -1 -1) (local.get $r) (f64x2.le (local.get $x) (v128.const f64x2 -3 -3))))
    (local.set $r
      (v128.bitselect (v128.const f64x2 1 1) (local.get $r) (f64x2.ge (local.get $x) (v128.const f64x2 3 3))))
    (v128.store64_lane 0 (local.get $p) (f32x4.demote_f64x2_zero (local.get $r))))

  ;; n float32 activations at $from -> float64 pairs at $to
  (func $splat (param $from i32) (param $to i32) (param $n i32)
    (loop $next
      (v128.store (local.get $to) (f64x2.splat (f64.promote_f32 (f32.load (local.get $from)))))
      (local.set $from (i32.add (local.get $from) (i32.const 4)))
      (local.set $to (i32.add (local.get $to) (i32.const 16)))
      (br_if $next (local.tee $n (i32.sub (local.get $n) (i32.const 1))))))

  (func (export "forward") (param $desc i32)
    (local $layers i32) (local $w i32) (local $nIn i32) (local $blocks i32) (local $out i32)
    (local $q i32) (local $i i32) (local $x v128)
    (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (local.set $layers (i32.load (local.get $desc)))
    (loop $layer
      (local.set $w (i32.load offset=4 (local.get $desc)))
      (local.set $nIn (i32.load offset=8 (local.get $desc)))
      (local.set $blocks (i32.load offset=12 (local.get $desc)))
      (local.set $out (i32.load offset=20 (local.get $desc)))
      (call $splat (i32.load offset=16 (local.get $desc)) (i32.const 1024) (local.get $nIn))
      (loop $block
        (local.set $a0 (v128.load (local.get $w)))
        (local.set $a1 (v128.load offset=16 (local.get $w)))
        (local.set $a2 (v128.load offset=32 (local.get $w)))
        (local.set $a3 (v128.load offset=48 (local.get $w)))
        (local.set $w (i32.add (local.get $w) (i32.const 64)))
        (local.set $q (i32.const 1024))
        (local.set $i (local.get $nIn))
        (loop $input
          (local.set $x (v128.load (local.get $q)))
          (local.set $a0 (f64x2.add (local.get $a0) (f64x2.mul (v128.load (local.get $w)) (local.get $x))))
          (local.set $a1 (f64x2.add (local.get $a1) (f64x2.mul (v128.load offset=16 (local.get $w)) (local.get $x))))
          (local.set $a2 (f64x2.add (local.get $a2) (f64x2.mul (v128.load offset=32 (local.get $w)) (local.get $x))))
          (local.set $a3 (f64x2.add (local.get $a3) (f64x2.mul (v128.load offset=48 (local.get $w)) (local.get $x))))
          (local.set $w (i32.add (local.get $w) (i32.const 64)))
          (local.set $q (i32.add (local.get $q) (i32.const 16)))
          (br_if $input (local.tee $i (i32.sub (local.get $i) (i32.const 1)))))
        (call $emit (local.get $out) (local.get $a0))
        (call $emit (i32.add (local.get $out) (i32.const 8)) (local.get $a1))
        (call $emit (i32.add (local.get $out) (i32.const 16)) (local.get $a2))
        (call $emit (i32.add (local.get $out) (i32.const 24)) (local.get $a3))
        (local.set $out (i32.add (local.get $out) (i32.const 32)))
        (br_if $block (local.tee $blocks (i32.sub (local.get $blocks) (i32.const 1)))))
      (local.set $desc (i32.add (local.get $desc) (i32.const 28)))
      (br_if $layer (local.tee $layers (i32.sub (local.get $layers) (i32.const 1))))))

  ;; both passes of a decision (the inputs and their mirror image) off a single load of each weight
  (func (export "forward2") (param $desc i32)
    (local $layers i32) (local $w i32) (local $nIn i32) (local $blocks i32) (local $out i32) (local $mout i32)
    (local $q i32) (local $i i32) (local $xa v128) (local $xb v128)
    (local $w0 v128) (local $w1 v128) (local $w2 v128) (local $w3 v128)
    (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (local $b0 v128) (local $b1 v128) (local $b2 v128) (local $b3 v128)
    (local.set $layers (i32.load (local.get $desc)))
    (loop $layer
      (local.set $w (i32.load offset=4 (local.get $desc)))
      (local.set $nIn (i32.load offset=8 (local.get $desc)))
      (local.set $blocks (i32.load offset=12 (local.get $desc)))
      (local.set $out (i32.load offset=20 (local.get $desc)))
      (local.set $mout (i32.load offset=28 (local.get $desc)))
      (call $splat (i32.load offset=16 (local.get $desc)) (i32.const 1024) (local.get $nIn))
      (call $splat (i32.load offset=24 (local.get $desc)) (i32.const 17408) (local.get $nIn))
      (loop $block
        (local.set $a0 (v128.load (local.get $w)))
        (local.set $a1 (v128.load offset=16 (local.get $w)))
        (local.set $a2 (v128.load offset=32 (local.get $w)))
        (local.set $a3 (v128.load offset=48 (local.get $w)))
        (local.set $b0 (local.get $a0))
        (local.set $b1 (local.get $a1))
        (local.set $b2 (local.get $a2))
        (local.set $b3 (local.get $a3))
        (local.set $w (i32.add (local.get $w) (i32.const 64)))
        (local.set $q (i32.const 1024))
        (local.set $i (local.get $nIn))
        (loop $input
          (local.set $xa (v128.load (local.get $q)))
          (local.set $xb (v128.load offset=16384 (local.get $q)))
          (local.set $w0 (v128.load (local.get $w)))
          (local.set $w1 (v128.load offset=16 (local.get $w)))
          (local.set $w2 (v128.load offset=32 (local.get $w)))
          (local.set $w3 (v128.load offset=48 (local.get $w)))
          (local.set $a0 (f64x2.add (local.get $a0) (f64x2.mul (local.get $w0) (local.get $xa))))
          (local.set $b0 (f64x2.add (local.get $b0) (f64x2.mul (local.get $w0) (local.get $xb))))
          (local.set $a1 (f64x2.add (local.get $a1) (f64x2.mul (local.get $w1) (local.get $xa))))
          (local.set $b1 (f64x2.add (local.get $b1) (f64x2.mul (local.get $w1) (local.get $xb))))
          (local.set $a2 (f64x2.add (local.get $a2) (f64x2.mul (local.get $w2) (local.get $xa))))
          (local.set $b2 (f64x2.add (local.get $b2) (f64x2.mul (local.get $w2) (local.get $xb))))
          (local.set $a3 (f64x2.add (local.get $a3) (f64x2.mul (local.get $w3) (local.get $xa))))
          (local.set $b3 (f64x2.add (local.get $b3) (f64x2.mul (local.get $w3) (local.get $xb))))
          (local.set $w (i32.add (local.get $w) (i32.const 64)))
          (local.set $q (i32.add (local.get $q) (i32.const 16)))
          (br_if $input (local.tee $i (i32.sub (local.get $i) (i32.const 1)))))
        (call $emit (local.get $out) (local.get $a0))
        (call $emit (i32.add (local.get $out) (i32.const 8)) (local.get $a1))
        (call $emit (i32.add (local.get $out) (i32.const 16)) (local.get $a2))
        (call $emit (i32.add (local.get $out) (i32.const 24)) (local.get $a3))
        (call $emit (local.get $mout) (local.get $b0))
        (call $emit (i32.add (local.get $mout) (i32.const 8)) (local.get $b1))
        (call $emit (i32.add (local.get $mout) (i32.const 16)) (local.get $b2))
        (call $emit (i32.add (local.get $mout) (i32.const 24)) (local.get $b3))
        (local.set $out (i32.add (local.get $out) (i32.const 32)))
        (local.set $mout (i32.add (local.get $mout) (i32.const 32)))
        (br_if $block (local.tee $blocks (i32.sub (local.get $blocks) (i32.const 1)))))
      (local.set $desc (i32.add (local.get $desc) (i32.const 28)))
      (br_if $layer (local.tee $layers (i32.sub (local.get $layers) (i32.const 1)))))))
