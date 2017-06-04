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
      .reg .u64   %cd<4>, %tidz;
      .reg .u32   %tidx, %tidy;
      .reg .<%= var(:f) %> %f<3>;
      .reg .pred  p;
      .reg .pred  first;

      ld.param.u64  %cd0, [pins];
      mov.u32       %tidx, %ctaid.z;
      mov.u32       %tidy, %ctaid.y;
      cvt.u64.u32   %tidz, %ctaid.x;

      // (%cd1) input.offset = input + (tid.x * sx + tid.y * sy * ix + tid.z * ix * iy) * float_size
      mul.wide.u32  %cd1, %tidx, <%= var(:sx) %>;
      mad.wide.u32  %cd1, %tidy, <%= var(:sy) * var(:x) %>, %cd1;
      mad.lo.u64    %cd1, %tidz, <%= var(:x) * var(:y) %>, %cd1;
      mad.lo.u64    %cd1, %cd1, <%= var(:float_size) %>, %cd0;
      <%= if pin_offset(:input) > 0 do %>
        add.u64     %cd1, %cd1, <%= pin_offset(:input) %>;
      <% end %>

      // (%cd2) output.offset = output + (tid.x + tid.y * ox + tid.z * ox * oy) * float_size
      cvt.u64.u32   %cd2, %tidx;
      mad.wide.u32  %cd2, %tidy, <%= var(:ox) %>, %cd2;
      mad.lo.u64    %cd2, %tidz, <%= var(:ox) * var(:oy) %>, %cd2;
      mad.lo.u64    %cd2, %cd2, <%= var(:float_size) %>, %cd0;
      <%= if pin_offset(:output) > 0 do %>
        add.u64     %cd2, %cd2, <%= pin_offset(:output) %>;
      <% end %>

      // %first - first item flag
      setp.eq.u32   first, 1, 1;
      // %tidy - py
      mov.u32       %tidy, <%= var(:py) %>;

    loop_y:
      mov.u32       %tidx, <%= var(:px) %>;
    loop_x:
      // acc = first ? [input] : max(acc, [input])
      ld.global.<%= var(:f) %> %f1, [%cd1];
      @!first max.<%= var(:f) %> %f0, %f0, %f1;
      @first  mov.<%= var(:f) %> %f0, %f1;
      @first  setp.eq.u32  first, 1, 0;
      // next point
      add.u64       %cd1, %cd1, <%= var(:float_size) %>;
      // count x
      sub.u32       %tidx, %tidx, 1;
      setp.ne.u32   p, %tidx, 0;
      @p bra        loop_x;
      // next line
      add.u64       %cd1, %cd1, <%= (var(:x) - var(:px)) * var(:float_size) %>;
      // count y
      sub.u32       %tidy, %tidy, 1;
      setp.ne.u32   p, %tidy, 0;
      @p bra        loop_y;

      st.global.<%= var(:f) %> [%cd2], %f0;
      ret;
    <% end %>
    """
  end

  defp back_ptx() do
    """
    <%= defkernel ctx, "back", shared: u64.ptr do %>
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
