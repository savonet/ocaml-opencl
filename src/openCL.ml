exception Error of int

let init () =
  Callback.register_exception "opencl_exn_error" (Error 0)

module Platform = struct
  type t

  external available : unit -> t array = "caml_opencl_platform_ids"

  type info = [`Profile | `Version | `Name | `Vendor | `Extensions]

  external info : t -> info -> string = "caml_opencl_platform_info"

  let profile p = info p `Profile
  let version p = info p `Version
  let name p = info p `Name
  let vendor p = info p `Vendor
  let extensions p = info p `Extensions
end

module Device = struct
(* type context_properties = [`Platform] *)

  type device_type = [`CPU | `GPU | `Accelerator | `Default | `All]

  type t
end

module Context = struct
  type t

  external create_from_type : Platform.t -> Device.device_type -> t = "caml_opencl_create_context_from_type"

  external devices : t -> Device.t array = "caml_opencl_context_devices"
end

module Program = struct
  type t

  external create_with_source : Context.t -> string -> t = "caml_opencl_create_program_with_source"

  let create_with_source_file c f =
    let src = ref "" in
    let ic = open_in f in
    (
      try
        while true do
          src := !src ^ input_line ic ^ "\n"
        done
      with
        | End_of_file -> ()
    );
    close_in ic;
    create_with_source c !src

  external build : t -> Device.t array -> string -> unit = "caml_opencl_build_program"
  let build p ?(options="") d = build p d options

  external build_log : t -> Device.t -> string = "caml_opencl_program_build_log"
end

module Buffer = struct
  type t

  type flag = [`Read_write | `Read_only | `Write_only]

  external create : Context.t -> flag array -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t = "caml_opencl_create_buffer"
  let create c f b = create c (Array.of_list f) b
end

module Kernel = struct
  type t

  external create : Program.t -> string -> t = "caml_opencl_create_kernel"

  external set_arg_int : t -> int -> int -> unit = "caml_opencl_set_kernel_arg_int"

  external set_arg_buffer : t -> int -> Buffer.t -> unit = "caml_opencl_set_kernel_arg_buffer"

  type argument = [ `Buffer of Buffer.t | `Int of int ]

  let set_args k (a:argument array) =
    for i = 0 to Array.length a - 1 do
      match a.(i) with
        | `Buffer b -> set_arg_buffer k i b
        | `Int n -> set_arg_int k i n
    done
end

module Event = struct
  type t

  external wait : t -> unit = "caml_opencl_wait_for_event"
end

module Command_queue = struct
  type t

  external create : Context.t -> Device.t -> t = "caml_opencl_create_command_queue"

  external finish : t -> unit = "caml_opencl_finish"

  external enqueue_nd_range_kernel : t -> Kernel.t -> int array -> int array -> Event.t = "caml_opencl_enqueue_nd_range_kernel"
end
