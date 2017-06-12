defmodule Neuro.Nodes.Pooling do
  alias Neuro.Nodes.Base
  alias Neuro.Nodes.Convolution
  use Base

  def __batch__(%{assigns: %{back_propagation: true, vars: vars}}) do
    [{:run, {"back", vars.block, vars.grid, []}}]
  end
  def __batch__(%{assigns: %{vars: vars}}) do
    [{:run, {"inference", vars.block, vars.grid, []}}]
  end

  def __ptx__(%{assigns: %{back_propagation: true}}) do
    back_ptx()
  end
  def __ptx__(_node) do
    inference_ptx()
  end

  defp inference_ptx() do
    """
    <%= defkernel ctx, "inference" do %>
      .reg .u64   %z, %x, %y, %i_ptr, %o_ptr;
      .reg .<%= var(:f) %> %i, %acc;
      .reg .pred  p;
      .reg .pred  first;

      ld.param.u64  %o_ptr, [pins];
      cvt.u64.u32   %x, %ctaid.z;
      cvt.u64.u32   %y, %ctaid.y;
      cvt.u64.u32   %z, %ctaid.x;

      // (%cd1) input.offset = input + (tid.x * sx + tid.y * sy * ix + tid.z * ix * iy) * float_size
      mul.lo.u64    %i_ptr, %x, <%= var(:sx) %>;
      mad.lo.u64    %i_ptr, %y, <%= var(:sy) * var(:x) %>, %i_ptr;
      mad.lo.u64    %i_ptr, %z, <%= var(:x) * var(:y) %>, %i_ptr;
      mad.lo.u64    %i_ptr, %i_ptr, <%= var(:float_size) %>, %o_ptr;
      <%= if pin_offset(:input) > 0 do %>
        add.u64     %i_ptr, %i_ptr, <%= pin_offset(:input) %>;
      <% end %>

      // (%cd2) output.offset = output + (tid.x + tid.y * ox + tid.z * ox * oy) * float_size
      mad.lo.u64    %x, %y, <%= var(:ox) %>, %x;
      mad.lo.u64    %x, %z, <%= var(:ox) * var(:oy) %>, %x;
      mad.lo.u64    %o_ptr, %x, <%= var(:float_size) %>, %o_ptr;
      <%= if pin_offset(:output) > 0 do %>
        add.u64     %o_ptr, %o_ptr, <%= pin_offset(:output) %>;
      <% end %>

      // %first - first item flag
      setp.eq.u32   first, 1, 1;

      mov.u64       %y, <%= var(:py) %>;
    loop_y:
      mov.u64       %x, <%= var(:px) %>;
    loop_x:
      // acc = first ? [input] : max(acc, [input])
      ld.global.<%= var(:f) %> %i, [%i_ptr];
      @!first max.<%= var(:f) %> %acc, %acc, %i;
      @first  mov.<%= var(:f) %> %acc, %i;
      @first  setp.eq.u32  first, 1, 0;
      // next point
      add.u64       %i_ptr, %i_ptr, <%= var(:float_size) %>;
      // count x
      sub.u64       %x, %x, 1;
      setp.ne.u64   p, %x, 0;
      @p bra        loop_x;
      // next line
      add.u64       %i_ptr, %i_ptr, <%= (var(:x) - var(:px)) * var(:float_size) %>;
      // count y
      sub.u64       %y, %y, 1;
      setp.ne.u64   p, %y, 0;
      @p bra        loop_y;

      st.global.<%= var(:f) %> [%o_ptr], %acc;
      ret;
    <% end %>
    """
  end

  defp back_ptx() do
    """
    <%= defkernel ctx, "back" do %>
      .shared .<%= var(:f) %> buffer[<%= var(:px) * var(:py) %>];
      .reg .u64   %z, %x, %y, %i_ptr, %o_ptr, %loss_ptr;
      .reg .<%= var(:f) %> %i, %acc, %loss;
      .reg .pred  p;
      .reg .pred  first;

      ld.param.u64  %loss_ptr, [pins];
      cvt.u64.u32   %x, %ctaid.z;
      cvt.u64.u32   %y, %ctaid.y;
      cvt.u64.u32   %z, %ctaid.x;

      // input.offset = (tid.x * sx + tid.y * sy * ix + tid.z * ix * iy) * float_size
      mul.lo.u64    %i_ptr, %x, <%= var(:sx) %>;
      mad.lo.u64    %i_ptr, %y, <%= var(:sy) * var(:x) %>, %i_ptr;
      mad.lo.u64    %i_ptr, %z, <%= var(:x) * var(:y) %>, %i_ptr;
      mad.lo.u64    %i_ptr, %i_ptr, <%= var(:float_size) %>, %loss_ptr;
      <%= if pin_offset(:inference) > 0 do %>
        add.u64     %i_ptr, %i_ptr, <%= pin_offset(:inference) %>;
      <% end %>

      // output.offset = (tid.x * sx + tid.y * sy * ix + tid.z * ix * iy) * float_size
      mul.lo.u64    %o_ptr, %x, <%= var(:sx) %>;
      mad.lo.u64    %o_ptr, %y, <%= var(:sy) * var(:x) %>, %o_ptr;
      mad.lo.u64    %o_ptr, %z, <%= var(:x) * var(:y) %>, %o_ptr;
      mad.lo.u64    %o_ptr, %o_ptr, <%= var(:float_size) %>, %loss_ptr;
      <%= if pin_offset(:input) > 0 do %>
        add.u64     %o_ptr, %o_ptr, <%= pin_offset(:input) %>;
      <% end %>

      // loss.offset = (tid.x + tid.y * ox + tid.z * ox * oy) * float_size
      mad.lo.u64    %x, %y, <%= var(:ox) %>, %x;
      mad.lo.u64    %x, %z, <%= var(:ox) * var(:oy) %>, %x;
      mad.lo.u64    %loss_ptr, %x, <%= var(:float_size) %>, %loss_ptr;
      <%= if pin_offset(:output) > 0 do %>
        add.u64     %loss_ptr, %loss_ptr, <%= pin_offset(:output) %>;
      <% end %>

      // find maximum inference value

      // %first - first item flag
      setp.eq.u32   first, 1, 1;
      mov.u64       %z, 0;
      mov.u64       %y, <%= var(:py) %>;
    loop_y:
      mov.u64       %x, <%= var(:px) %>;
    loop_x:
      // acc = first ? [input] : max(acc, [input])
      ld.global.<%= var(:f) %>    %i, [%i_ptr];
      st.shared.<%= var(:f) %>    buffer[%z], %i;
      @!first max.<%= var(:f) %>  %acc, %acc, %i;
      @first  mov.<%= var(:f) %>  %acc, %i;
      @first  setp.eq.u32  first, 1, 0;
      // next point
      add.u64       %i_ptr, %i_ptr, <%= var(:float_size) %>;
      add.u64       %z, %z, 1;
      // count x
      sub.u64       %x, %x, 1;
      setp.ne.u64   p, %x, 0;
      @p bra        loop_x;
      // next line
      add.u64       %i_ptr, %i_ptr, <%= (var(:x) - var(:px)) * var(:float_size) %>;
      // count y
      sub.u64       %y, %y, 1;
      setp.ne.u64   p, %y, 0;
      @p bra        loop_y;

      // fill output
      ld.global.<%= var(:f) %> %loss, [%loss_ptr];
      mov.u64       %z, 0;
      mov.u64       %y, <%= var(:py) %>;
    loss_loop_y:
      mov.u64       %x, <%= var(:px) %>;
    loss_loop_x:
      ld.shared.<%= var(:f) %>      %i, buffer[%z];
      setp.eq.<%= var(:f) %>        p, %acc, %i;
      @p st.global.<%= var(:f) %>   [%o_ptr], %loss;
      @!p st.global.<%= var(:f) %>  [%o_ptr], 0.0;
      // next point
      add.u64       %o_ptr, %o_ptr, <%= var(:float_size) %>;
      add.u64       %z, %z, 1;
      // count x
      sub.u64       %x, %x, 1;
      setp.ne.u64   p, %x, 0;
      @p bra        loss_loop_x;
      // next line
      add.u64       %o_ptr, %o_ptr, <%= (var(:x) - var(:px)) * var(:float_size) %>;
      // count y
      sub.u64       %y, %y, 1;
      setp.ne.u64   p, %y, 0;
      @p bra        loss_loop_y;

      ret;
    <% end %>
    """
  end

  def vars(opts, %{gpu_info: info}) do
    {x, y, z} =  opts |> Keyword.get(:size) |> Base.triple_size()
    {px, py} =   opts |> Keyword.get(:pooling) |> get_pooling_size()
    {sx, sy} =   opts |> Keyword.get(:stride) |> Base.stride()

    ox = round((x - px + sx) / sx)
    oy = round((y - py + sy) / sy)
    oz = z

    {block, grid} = Convolution.cta(ox, oy, oz, info)

    %{x: x, y: y, z: z,
      ox: ox, oy: oy, oz: oz,
      px: px, py: py,
      sx: sx, sy: sy,
      grid: grid, block: block}
  end

  defp get_pooling_size({_, _} = tuple), do: tuple
  defp get_pooling_size(x), do: {x, x}
end
