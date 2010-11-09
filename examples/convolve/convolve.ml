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
    b_in.{i} <- (Random.float (float max_int))
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
          sum := !sum +. b_filter.{r * filter_width + c} *. b_in.{y * in_width + x}
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
    if b_out.{i} <> b_out_cpu.{i} then
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
  let src =
    let ans = ref "" in
    let ic = open_in "kernels.cl" in
    (
    try
      while true do
        ans := !ans ^ input_line ic ^ "\n"
      done
    with
      | End_of_file -> ()
    );
    close_in ic;
    !ans
  in
  (* Printf.printf ">>> Prog\n%s<<<Prog\n%!" src; *)
  let prog = OpenCL.Program.create_with_source ctxt src in
  (
    try
      OpenCL.Program.build prog [|dev|] "";
    with
    | e ->
      Printf.printf "ERROR:\n%s\n%!" (OpenCL.Program.build_log prog dev);
      raise e
  );
  Printf.printf "Program built\n%!";
  let kernels = Array.map (fun kn -> OpenCL.Kernel.create prog kn) kernels in
  let kernel = kernels.(0) in
  init_data ();
  let b_in = OpenCL.Buffer.create ctxt [`Read_only] b_in in
  let b_out = OpenCL.Buffer.create ctxt [`Write_only] b_out in
  let b_filter = OpenCL.Buffer.create ctxt [`Read_only] b_filter in
  OpenCL.Kernel.set_arg_buffer kernel 0 b_in;
  OpenCL.Kernel.set_arg_buffer kernel 1 b_filter;
  OpenCL.Kernel.set_arg_buffer kernel 2 b_out;
  OpenCL.Kernel.set_arg_int kernel 3 in_width;
  OpenCL.Kernel.set_arg_int kernel 4 filter_width;
  OpenCL.Command_queue.finish queue;
  Printf.printf "Compute using CL ... %!";
  let t = Sys.time () in
  let event = OpenCL.Command_queue.enqueue_nd_range_kernel queue kernel [|width; height|] [|32; 32|] in
  OpenCL.Event.wait event;
  let t = Sys.time () -. t in
  Printf.printf "done (%.02fs)\n%!" t;
  OpenCL.Command_queue.finish queue;
  ignore [b_in; b_out; b_filter];
  cpu_compute ();
  check ();
  List.iter OpenCL.Buffer.release [b_in; b_out; b_filter];
  ()
