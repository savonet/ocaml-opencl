(** Error executing a function. *)
exception Error of int

(** Initialize the openCL library. This function should be called once, before
    using any other function. *)
val init : unit -> unit

val unload_compiler : unit -> unit

(** Operations on platforms. *)
module Platform : sig
  (** A platform. *)
  type t

  (** List available platforms. *)
  val available : unit -> t array

  val profile : t -> string

  val version : t -> string

  val name : t -> string

  val vendor : t -> string

  val extensions : t -> string
end

module Device : sig
  type device_type = [ `Accelerator | `All | `CPU | `Default | `GPU ]

  type t
end

module Context : sig
  type t

  val create_from_type : ?platform:Platform.t -> Device.device_type -> t

  (** List devices available on a platform. *)
  val devices : t -> Device.t array
end

module Program : sig
  type t

  val create_with_source_file : Context.t -> string -> t

  val create_with_source : Context.t -> string -> t

  val build : ?options:string -> ?devices:Device.t array -> t -> unit

  val build_log : t -> Device.t -> string
end

module Buffer : sig
  type t

  type flag = [ `Read_only | `Read_write | `Write_only | `Alloc_device ]

  val create : Context.t -> flag list -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
end

module Kernel : sig
  type t

  val create : Program.t -> string -> t

  type argument = [ `Buffer of Buffer.t | `Int of int ]

  val set_args : t -> argument array -> unit

  val set_arg_int : t -> int -> int -> unit

  val set_arg_buffer : t -> int -> Buffer.t -> unit
end

module Event : sig
  type t

  (** Wait for an event to be completed. *)
  val wait : t -> unit

  val duration : t -> Int64.t
end

module Command_queue : sig
  type t

  val create : Context.t -> Device.t -> t

  val finish : t -> unit

  val nd_range_kernel : t -> Kernel.t -> ?local_work_size:int array -> int array -> Event.t

  val read_buffer : t -> Buffer.t -> bool -> int -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> Event.t
end

(** Helper function to quickly test a kernel. *)
val run : ?platform:Platform.t -> ?device_type:Device.device_type -> string -> ?build_options:string ->
  string ->
  [ `Buffer_in of ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  | `Buffer_out of ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  | `Int of int ] array ->
  ?local_work_size:int array -> int array -> unit
