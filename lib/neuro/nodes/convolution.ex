defmodule Neuro.Nodes.Convolution do
  alias Neuro.Nodes.Base
  use Base

  def __batch__(%{assigns: %{back_propagation: true, vars: vars}}) do
    [{:run, {"back", vars.block, vars.grid, [:shared]}}]
  end
  def __batch__(%{assigns: %{vars: vars}}) do
    [{:run, {"inference", vars.block, vars.grid, [:shared]}}]
  end

  def __ptx__(%{assigns: %{back_propagation: true}}) do
    back_ptx()
  end
  def __ptx__(_node) do
    inference_ptx()
  end

  defp inference_ptx() do
    """
    <%= defkernel ctx, "inference", shared: u64.ptr do %>
      .reg .u64   %x, %y, %z, %pins, %i_ptr, %o_ptr, %w_ptr, %cx, %cy;
      .reg .<%= var(:f) %> %acc, %i, %w;
      .reg .pred  p;
      <%= if training?() do %>
        .reg .u64 %states_ptr;
      <% end %>

      ld.param.u64  %pins, [pins];
      ld.param.u64  %w_ptr, [shared];
      cvt.u64.u32   %x, %ctaid.z;
      cvt.u64.u32   %y, %ctaid.y;
      cvt.u64.u32   %z, %ctaid.x;

      // input.offset = input + (tid.x * sx + tid.y * sy * x) * float_size
      mul.lo.u64    %i_ptr, %x, <%= var(:sx) %>;
      mad.lo.u64    %i_ptr, %y, <%= var(:sy) * var(:x) %>, %i_ptr;
      mad.lo.u64    %i_ptr, %i_ptr, <%= var(:float_size) %>, %pins;
      <%= if pin_offset(:input) > 0 do %>
        add.u64       %i_ptr, %i_ptr, <%= pin_offset(:input) %>;
      <% end %>

      // output.offset = output + (tid.z * ox * oy + tid.y * ox + tid.x) * float_size
      mad.lo.u64    %o_ptr, %y, <%= var(:ox) %>, %x;
      mad.lo.u64    %o_ptr, %z, <%= var(:ox) * var(:oy) %>, %o_ptr;
      <%= if training?() do %>
        mad.lo.u64    %states_ptr, %o_ptr, <%= var(:float_size) %>, %w_ptr;
        <%= if shared_offset(:states) > 0 do %>
          add.u64     %states_ptr, %states_ptr, <%= shared_offset(:states) %>;
        <% end %>
      <% end %>
      mad.lo.u64    %o_ptr, %o_ptr, <%= var(:float_size) %>, %pins;
      <%= if pin_offset(:output) > 0 do %>
        add.u64     %o_ptr, %o_ptr, <%= pin_offset(:output) %>;
      <% end %>

      // w.offset = w + tid.z * wx * wy * float_size
      mad.lo.u64    %w_ptr, %z, <%= var(:wx) * var(:wy) * var(:float_size) %>, %w_ptr;
      <%= if shared_offset(:weights) > 0 do %>
        add.u64     %w_ptr, %w_ptr, <%= shared_offset(:weights) %>;
      <% end %>

      // %acc - accumulator
      mov.f32       %acc, 0.0;
      // %cy = wy
      mov.u64       %cy, <%= var(:wy) %>;

    loop_y:
      mov.u64       %cx, <%= var(:wx) %>;
    loop_x:
      // acc = acc + [input] * [w]
      ld.global.<%= var(:f) %> %w, [%w_ptr];
      ld.global.<%= var(:f) %> %i, [%i_ptr];
      mad.rn.<%= var(:f) %>    %acc, %w, %i, %acc;
      // next point
      add.u64       %w_ptr, %w_ptr, <%= var(:float_size) %>;
      add.u64       %i_ptr, %i_ptr, <%= var(:float_size) %>;
      // count x
      sub.u64       %cx, %cx, 1;
      setp.ne.u64   p, %cx, 0;
      @p bra        loop_x;
      // next line
      add.u64       %i_ptr, %i_ptr, <%= (var(:x) - var(:wx)) * var(:float_size) %>;
      // count y
      sub.u64       %cy, %cy, 1;
      setp.ne.u64   p, %cy, 0;
      @p bra        loop_y;

      <%= if training?() do %>
        st.global.<%= var(:f) %> [%states_ptr], %acc;
      <% end %>

      <%= include ctx, var(ctx, :activation), in: "acc", pred: "p", f: var(:f) %>
      st.global.<%= var(:f) %> [%o_ptr], %acc;
      ret;
    <% end %>
    """
  end

  defp back_ptx() do
    """
    <%= defkernel ctx, "back", shared: u64.ptr do %>
      .reg .u64   %x, %y, %z, %loss_ptr, %states_ptr, %o_ptr, %w_ptr, %tmp,
                  %cx, %cy, %cz, %lx, %ly, %w_base, %loss_base, %states_base;
      .reg .<%= var(:f) %> %acc, %loss, %w, %activation;
      .reg .pred  p;

      ld.param.u64  %loss_ptr, [pins];
      ld.param.u64  %w_ptr, [shared];
      cvt.u64.u32   %x, %ctaid.z;
      cvt.u64.u32   %y, %ctaid.y;
      cvt.u64.u32   %z, %ctaid.x;

      // output.offset = output + (tid.x * sx + tid.y * sy * x + tid.z * x * y) * float_size
      mul.lo.u64    %o_ptr, %x, <%= var(:sx) %>;
      mad.lo.u64    %o_ptr, %y, <%= var(:sy) * var(:x) %>, %o_ptr;
      mad.lo.u64    %o_ptr, %z, <%= var(:x) * var(:y) %>, %o_ptr;
      mad.lo.u64    %o_ptr, %o_ptr, <%= var(:float_size) %>, %loss_ptr;
      <%= if pin_offset(:input) > 0 do %>
        add.u64     %o_ptr, %o_ptr, <%= pin_offset(:input) %>;
      <% end %>

      // loss.offset = input + (tid.x + tid.y * ox) * float_size
      mad.lo.u64    %cx, %y, <%= var(:ox) %>, %x;
      mad.lo.u64    %loss_ptr, %cx, <%= var(:float_size) %>, %loss_ptr;
      <%= if pin_offset(:output) > 0 do %>
        add.u64     %loss_ptr, %loss_ptr, <%= pin_offset(:output) %>;
      <% end %>

      mad.lo.u64    %states_ptr, %cx, <%= var(:float_size) %>, %w_ptr;
      <%= if shared_offset(:states) > 0 do %>
        add.u64     %states_ptr, %states_ptr, <%= shared_offset(:states) %>;
      <% end %>
      <%= if shared_offset(:weights) > 0 do %>
        add.u64     %w_ptr, %w_ptr, <%= shared_offset(:weights) %>;
      <% end %>

      mov.<%= var(:f) %>    %acc, 0.0;
      mov.u64               %cz, 0;
      mov.u64               %w_base, %w_ptr;
      mov.u64               %loss_base, %loss_ptr;
      mov.u64               %states_base, %states_ptr;

    z_loop:
      mov.u64       %cy, 0;
      mov.u64       %ly, %y;

    y_loop:
      setp.hi.u64   p, %ly, <%= var(:oy) - 1 %>;
      @p bra        skip_y;
      mov.u64       %cx, 0;
      mov.u64       %lx, %x;

    x_loop:
      setp.hi.u64   p, %lx, <%= var(:ox) - 1 %>;
      @p bra        skip_x;

      ld.global.<%= var(:f) %> %activation, [%states_ptr];
      <%= include ctx, var(:activation), :back_propagation, in: "activation", pred: "p", f: var(:f) %>
      setp.eq.<%= var(:f) %> p, %activation, 0.0;
      @p bra        skip_x;

      ld.global.<%= var(:f) %> %w, [%w_ptr];
      ld.global.<%= var(:f) %> %loss, [%loss_ptr];
      mul.rn.<%= var(:f) %>    %activation, %activation, %w;
      mad.rn.<%= var(:f) %>    %acc, %activation, %loss, %acc;

    skip_x:
      setp.eq.u64   p, %lx, 0;
      @p bra        skip_y;
      sub.u64       %lx, %lx, 1;
      add.u64       %cx, %cx, 1;
      setp.hi.u64   p, %cx, <%= var(:wx) - 1 %>;
      @p bra        skip_y;
      add.u64       %w_ptr, %w_ptr, <%= var(:float_size) %>;
      sub.u64       %loss_ptr, %loss_ptr, <%= var(:float_size) %>;
      sub.u64       %states_ptr, %states_ptr, <%= var(:float_size) %>;
      bra           x_loop;

    skip_y:
      setp.eq.u64   p, %ly, 0;
      @p bra        skip_z;
      sub.u64       %ly, %ly, 1;
      add.u64       %cy, %cy, 1;
      setp.hi.u64   p, %cy, <%= var(:wy) - 1 %>;
      @p bra        skip_z;
      mad.lo.u64    %w_ptr, %cy, <%= var(:wx) * var(:float_size) %>, %w_base;
      sub.u64       %tmp, %y, %ly;
      mul.lo.u64    %tmp, %tmp, <%= var(:ox) * var(:float_size) %>;
      sub.u64       %loss_ptr, %loss_base, %tmp;
      sub.u64       %states_ptr, %states_base, %tmp;
      bra           y_loop;

    skip_z:
      add.u64       %cz, %cz, 1;
      setp.hi.u64   p, %cz, <%= var(:wz) - 1 %>;
      @p bra        break;
      mad.lo.u64    %w_base, %cz, <%= var(:wx) * var(:wz) * var(:float_size) %>, %w_base;
      mov.u64       %w_ptr, %w_base;
      mad.lo.u64    %loss_base, %cz, <%= var(:ox) * var(:oy) * var(:float_size) %>, %loss_base;
      mov.u64       %loss_ptr, %loss_base;
      mad.lo.u64    %states_base, %cz, <%= var(:ox) * var(:oy) * var(:float_size) %>, %states_base;
      mov.u64       %states_ptr, %states_base;
      bra           z_loop;

    break:
      st.global.<%= var(:f) %> [%o_ptr], %acc;

      ret;
    <% end %>
    """
  end

  def vars(opts, %{gpu_info: info}) do
    {x, y, z}    = opts |> Keyword.get(:size) |> Base.triple_size()
    {wx, wy, wz} = opts |> Keyword.get(:kernel_size) |> Base.triple_size()
    {sx, sy}     = opts |> Keyword.get(:stride) |> Base.stride()
    activation   = opts |> Keyword.get(:activation, :relu) |> Base.activation()

    ox = round((x - wx + sx) / sx)
    oy = round((y - wy + sy) / sy)
    oz = wz

    {block, grid} = if opts[:back_propagation] do
      cta(x, y, z, info)
    else
      cta(ox, oy, oz, info)
    end

    %{x: x, y: y, z: z,
      ox: ox, oy: oy, oz: oz,
      wx: wx, wy: wy, wz: wz,
      sx: sx, sy: sy,
      grid: grid, block: block,
      activation: activation}
  end

  def cta(ox, oy, oz, info) do
    {max_z, max_y, max_x} = info[:max_grid]
    if ox > max_x do
      raise RuntimeError, message: "Maximum allowed layer width is #{max_x}"
    end
    if oy > max_y do
      raise RuntimeError, message: "Maximum allowed layer height is #{max_y}"
    end
    if oz > max_z do
      raise RuntimeError, message: "Maximum allowed layer depth is #{max_z}"
    end
    {{1, 1, 1}, {oz, oy, ox}}
  end

  def shared(key, vars) do
    shared = %{weights: %{key => {vars.f, vars.wx * vars.wy * vars.wz}},
               biases:  %{key => {vars.f, vars.wx * vars.wy * vars.wz}}}
    shared = if vars.training do
      Map.merge(shared, %{states: %{key => {vars.f, vars.ox * vars.oy * vars.oz}}})
    else
      shared
    end
    %{shared: shared}
  end
end
