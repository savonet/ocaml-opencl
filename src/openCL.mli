exception Error of int

val init : unit -> unit

module Platform : sig
  type t

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

  val create_from_type : Platform.t -> Device.device_type -> t

  val devices : t -> Device.t array
end

module Program : sig
  type t

  val create_with_source : Context.t -> string -> t

  val build : t -> Device.t array -> string -> unit

  val build_log : t -> Device.t -> string
end

module Buffer : sig
  type t

  type flag = [ `Read_only | `Read_write | `Write_only ]

  val create : Context.t -> flag list -> (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t -> t
end

module Kernel : sig
  type t

  val create : Program.t -> string -> t

  val set_arg_int : t -> int -> int -> unit

  val set_arg_buffer : t -> int -> Buffer.t -> unit
end

module Event : sig
  type t

  val wait : t -> unit
end

module Command_queue : sig
  type t

  val create : Context.t -> Device.t -> t

  val finish : t -> unit

  val enqueue_nd_range_kernel : t -> Kernel.t -> int array -> int array -> Event.t
end
