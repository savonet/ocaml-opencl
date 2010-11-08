let () =
  Printf.printf "Platforms: %d\n%!" (OpenCL.num_platforms ());
  let ids = OpenCL.platform_ids () in
  let id = ids.(0) in
  Printf.printf "%s\n%!" (OpenCL.platform_info id `Name);
  let ctxt = OpenCL.create_context_from_type id `CPU in
  let devs = OpenCL.context_devices ctxt in
  let dev = devs.(0) in
  let queue = OpenCL.create_command_queue ctxt dev in
  ()
