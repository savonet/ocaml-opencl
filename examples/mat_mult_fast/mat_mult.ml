let m = 256
let n = 512
let p = 1024

let a = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (m * n)
let b = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (n * p)
let cpu = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (m * p)
let gpu = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (m * p)

let () = Random.self_init ()

let randomize a =
  Printf.printf "Randomizing... %!";
  for i = 0 to Bigarray.Array1.dim a - 1 do
    a.{i} <- (* Random.float *) 1.
  done;
  Printf.printf "done\n%!"

let cpu_compute () =
  Printf.printf "Computing using CPU... %!";
  let t = Sys.time () in
  for i = 0 to m - 1 do
    for j = 0 to p - 1 do
      let sum = ref 0. in
      for k = 0 to n - 1 do
        sum := !sum +. a.{i * n + k} *. b.{k * p + j}
      done;
      cpu.{i * p + j} <- !sum;
    done;
  done;
  let t = Sys.time () -.  t in
  Printf.printf "done (%.02fs)\n%!" t

let check () =
  Printf.printf "Compare CL and CPU results... %!";
  for i = 0 to Bigarray.Array1.dim cpu - 1 do
    if abs_float (gpu.{i} -. cpu.{i}) >= 0.001 then
      Printf.printf "Mismatch at %d: %.04f instead of %.04f\n%!" i gpu.{i} cpu.{i}
  done;
  Printf.printf "done\n%!"

let () =
  OpenCL.init ();
  randomize a;
  randomize b;
  Printf.printf "Computing using CL ... %!";
  let t = Sys.time () in
  OpenCL.run ~device_type:`CPU "kernels.cl" "Mat_mult" [|`Buffer_in a; `Buffer_in b; `Buffer_out gpu; `Int n; `Int p|] [|m;p|] ~local_work_size:[|32;32|];
  let t = Sys.time () -. t in
  Printf.printf "done (%.02fs)\n%!" t;
  cpu_compute ();
  check ();
  Gc.full_major ()
