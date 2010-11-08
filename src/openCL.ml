external num_platforms : unit -> int = "caml_opencl_num_platforms"

type platform_id

external platform_ids : unit -> platform_id array = "caml_opencl_platform_ids"

type platform_info = [`Profile | `Version | `Name | `Vendor | `Extensions]

external platform_info : platform_id -> platform_info -> string = "caml_opencl_platform_info"

let platform_profile p = platform_info p `Profile
let platform_version p = platform_info p `Version
let platform_name p = platform_info p `Name
let platform_vendor p = platform_info p `Vendor
let platform_extensions p = platform_info p `Extensions

(* type context_properties = [`Platform] *)

type device_type = [`CPU | `GPU | `Accelerator | `Default | `All]

type context

external create_context_from_type : platform_id -> device_type -> context = "caml_opencl_create_context_from_type"

type device_id

external context_devices : context -> device_id array = "caml_opencl_context_devices"

type command_queue

external create_command_queue : context -> device_id -> command_queue = "caml_opencl_create_command_queue"

type program

external create_program_with_source : context -> string -> program = "caml_opencl_create_program_with_source"

external build_program : program -> device_id array -> string -> unit = "caml_opencl_build_program"

type kernel

external create_kernel : program -> string -> kernel = "caml_opencl_create_kernel"
