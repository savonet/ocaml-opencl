let () =
  OpenCL.init ();
  let ids = OpenCL.Platform.available () in
  let id = ids.(0) in
  Printf.printf "%s\n%!" (OpenCL.Platform.info id `Name);
  let ctxt = OpenCL.Context.create_from_type id `CPU in
  let devs = OpenCL.Context.devices ctxt in
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
    | _ ->
      Printf.printf "ERROR:\n%s\n%!" (OpenCL.Program.build_log prog dev)
  );
  ()
