defmodule Neuro.Nodes.Pooling do
  alias Neuro.Nodes.Base
  use Base

  def __ptx__(_node) do
    """
    .version 5.0
    .target sm_30
    .address_size 64

    <%= defkernel ctx, "pooling" do %>
      .reg .u64   %cd<4>;
      .reg .u32   %c<7>;
      .reg .u64   %vd<3>;
      .reg .u32   %v<1>;
      .reg .<%= var(ctx, :f) %> %f<3>;
      .reg .pred  p;
      .reg .pred  first;

      ld.param.u64  %cd0, [pins];
      // ld.param.u64  %cd1, [weights];
      mov.u32       %c2, %tid.x;
      mov.u32       %c3, %tid.y;
      mov.u32       %c4, %tid.z;
      mov.u32       %c5, %ntid.x;
      mov.u32       %c6, %ntid.y;

      // (%cd2) ninput.x = (ntid.x - 1) * sx + px
      sub.u32       %v0, %c5, 1;
      mad.wide.u32  %cd3, %v0, <%= var(ctx, :sx) %>, <%= var(ctx, :px) %>;

      // (%cd1) input.offset = input + (tid.x * sx + tid.y * sy * ninput.x) * float_size
      mul.wide.u32  %vd0, %c3, <%= var(ctx, :sy) %>;
      mul.lo.u64    %vd0, %vd0, %cd2;
      mad.wide.u32  %vd0, %c2, <%= var(ctx, :sx) %>, %vd0;
      mad.lo.u64    %cd1, %vd0, <%= var(ctx, :float_size) %>, %cd0;
      add.u64       %cd1, %cd1, <%= offset(ctx, :input) %>;

      // (%cd2) output.offset = output + (tid.z * ntid.x * ntid.y + tid.y * ntid.x + tid.x) * float_size
      cvt.u64.u32   %vd1, %c2;
      mad.wide.u32  %vd0, %c3, %c5, %vd1;
      mul.wide.u32  %vd1, %c5, %c6;
      cvt.u64.u32   %vd2, %c4;
      mad.lo.u64    %vd0, %vd1, %vd2, %vd0;
      mad.lo.u64    %cd2, %vd0, <%= var(ctx, :float_size) %>, %cd0;
      add.u64       %cd2, %cd2, <%= offset(ctx, :output) %>;

      // (%vd0) dx = (ninput.x - px) * float_size
      sub.u64       %vd0, %cd3, <%= var(ctx, :px) %>;
      mul.lo.u64    %vd0, %vd0, <%= var(ctx, :float_size) %>;

      // %first - first item flag
      setp.eq.u32   first, 1, 1;
      // %c1 - wy
      mov.u32       %c1, <%= var(ctx, :py) %>;

    loop_y:
      mov.u32       %v0, <%= var(ctx, :px) %>;
    loop_x:
      // acc = first ? [input] : max(acc, [input])
      ld.global.<%= var(ctx, :f) %> %f1, [%cd1];
      @!first max.<%= var(ctx, :f) %> %f0, %f0, %f1;
      @first  mov.<%= var(ctx, :f) %> %f0, %f1;
      @first  setp.eq.u32  first, 1, 0;
      // next point
      add.u64       %cd1, %cd1, <%= var(ctx, :float_size) %>;
      // count x
      sub.u32       %v0, %v0, 1;
      setp.ne.u32   p, %v0, 0;
      @p bra        loop_x;
      // next line
      add.u64       %cd1, %cd1, %vd0;
      // count y
      sub.u32       %c1, %c1, 1;
      setp.ne.u32   p, %c1, 0;
      @p bra        loop_y;

      st.global.<%= var(ctx, :f) %> [%cd2], %f0;
      ret;
    <% end %>
    """
  end

  def vars(opts) do
    {x, y, z} =  opts |> Map.get(:size) |> Base.triple_size()
    {px, py} =   opts |> Map.get(:pooling_size) |> get_pooling_size()
    {sx, sy} =   opts |> Map.get(:stride) |> Base.stride()
    float_size = opts |> Map.get(:float_size) |> Base.float_size()
    f = "f#{float_size * 8}"

    ox = round((x - px + sx) / sx)
    oy = round((y - py + sy) / sy)
    oz = z

    %{x: x, y: y, z: z,
      ox: ox, oy: oy, oz: oz,
      px: px, py: py,
      sx: sx, sy: sy,
      f: f, float_size: float_size}
  end

  defp get_pooling_size({_, _} = tuple), do: tuple
  defp get_pooling_size(x), do: {x, x}
end
