let kernels = [|"Convolve"|]
let coef = 2
let width = 1024*coef
let height = 1024*coef
let filter_width = 17
let in_width = width + filter_width - 1
let in_height = height + filter_width - 1

let b_in = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (in_width * in_height)
let b_out_cpu = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (width * height)
let b_out = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (width * height)
let b_filter = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (filter_width * filter_width)

let init_data () =
  Printf.printf "Initializing data... %!";
  Random.self_init ();
  for i = 0 to Bigarray.Array1.dim b_in - 1 do
    b_in.{i} <- Random.float 1.
  done;
  for i = 0 to Bigarray.Array1.dim b_filter - 1 do
    b_filter.{i} <- Random.float 1.
  done;
  for i = 0 to Bigarray.Array1.dim b_out - 1 do
    b_out.{i} <- 666.
  done;
  Printf.printf "done\n%!"

let cpu_compute () =
  Printf.printf "Compute using CPU... %!";
  let t = Sys.time () in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let sum = ref 0. in
      for r = 0 to filter_width - 1 do
        for c = 0 to filter_width - 1 do
          sum := !sum +. b_filter.{r * filter_width + c} *. b_in.{y * in_width + x + c}
        done
      done;
      b_out_cpu.{y * width + x} <- !sum
    done
  done;
  let t = Sys.time () -.  t in
  Printf.printf "done (%.02fs)\n%!" t

let check () =
  Printf.printf "Compare CL and CPU results... %!";
  for i = 0 to Bigarray.Array1.dim b_out - 1 do
    if abs_float (b_out.{i} -. b_out_cpu.{i}) >= 0.0001 then
      Printf.printf "Mismatch at %d: %.04f vs %.04f\n%!" i b_out.{i} b_out_cpu.{i}
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
  let kernel = OpenCL.Kernel.create prog "Convolve" in
  init_data ();
  let b_in = OpenCL.Buffer.create ctxt [`Read_only] b_in in
  let b_out = OpenCL.Buffer.create ctxt [`Write_only] b_out in
  let b_filter = OpenCL.Buffer.create ctxt [`Read_only] b_filter in
  OpenCL.Kernel.set_args kernel [|`Buffer b_in; `Buffer b_filter; `Buffer b_out; `Int in_width; `Int filter_width|];
  OpenCL.Command_queue.finish queue;
  Printf.printf "Compute using CL ... %!";
  let t = Sys.time () in
  let event = OpenCL.Command_queue.enqueue_nd_range_kernel queue kernel [|width; height|] ~local_work_size:[|32; 32|] in
  OpenCL.Event.wait event;
  let t = Sys.time () -. t in
  Printf.printf "done (%.02fs)\n%!" t;
  OpenCL.Command_queue.finish queue;
  cpu_compute ();
  check ();
  Gc.full_major ()
