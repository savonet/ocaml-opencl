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
    a.{i} <- Random.float 1.
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
  let ids = OpenCL.Platform.available () in
  Printf.printf "OpenCL: %d platform(s) available\n%!" (Array.length ids);
  let id = ids.(0) in
  Printf.printf "Platform 0:\n - %s\n - %s\n - %s\n - %s\n - %s\n%!" (OpenCL.Platform.profile id) (OpenCL.Platform.version id) (OpenCL.Platform.name id) (OpenCL.Platform.vendor id) (OpenCL.Platform.extensions id);
  let ctxt = OpenCL.Context.create_from_type id `CPU in
  let devs = OpenCL.Context.devices ctxt in
  Printf.printf "CPU: %d device(s) available\n%!" (Array.length devs);
  let dev = devs.(0) in
  let queue = OpenCL.Command_queue.create ctxt dev in
  let prog = OpenCL.Program.create_with_source_file ctxt "kernels.cl" in
  (
    try
      OpenCL.Program.build prog [|dev|];
    with
    | e ->
      Printf.printf "ERROR:\n%s\n%!" (OpenCL.Program.build_log prog dev);
      raise e
  );
  Printf.printf "Program built\n%!";
  let kernel = OpenCL.Kernel.create prog "Mat_mult" in
  randomize a;
  randomize b;
  let a = OpenCL.Buffer.create ctxt [`Read_only] a in
  let b = OpenCL.Buffer.create ctxt [`Read_only] b in
  let gpu = OpenCL.Buffer.create ctxt [`Write_only] gpu in
  OpenCL.Kernel.set_args kernel [|`Buffer a; `Buffer b; `Buffer gpu; `Int n; `Int p|];
  OpenCL.Command_queue.finish queue;
  Printf.printf "Compute using CL ... %!";
  let t = Sys.time () in
  let event = OpenCL.Command_queue.enqueue_nd_range_kernel queue kernel [|m; p|] ~local_work_size:[|32; 32|] in
  OpenCL.Event.wait event;
  let t = Sys.time () -. t in
  Printf.printf "done (%.02fs)\n%!" t;
  OpenCL.Command_queue.finish queue;
  cpu_compute ();
  check ();
  Gc.full_major ()
