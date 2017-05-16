defmodule Neuro.Nodes.Convolution do
  alias Neuro.Nodes.Base
  use Base

  def __batch__(%{assigns: %{vars: vars}}) do
    [{"conv", {vars.ox, vars.oy, vars.oz}, {1, 1, 1}, [:w]}]
  end

  def __ptx__(_node) do
    """
    .version 5.0
    .target sm_30
    .address_size 64

    <%= defkernel ctx, "conv", weights: u64.ptr do %>
      .reg .u64   %cd<5>;
      .reg .u32   %c<7>;
      .reg .u64   %vd<3>;
      .reg .u32   %v<1>;
      .reg .<%= var(ctx, :f) %> %f<3>;
      .reg .pred  p;

      ld.param.u64  %cd0, [pins];
      ld.param.u64  %cd1, [weights];
      mov.u32       %c2, %tid.x;
      mov.u32       %c3, %tid.y;
      mov.u32       %c4, %tid.z;
      mov.u32       %c5, %ntid.x;
      mov.u32       %c6, %ntid.y;

      // (%cd4) ninput.x = (ntid.x - 1) * sx + wx
      sub.u32       %v0, %c5, 1;
      mad.wide.u32  %cd4, %v0, <%= var(ctx, :sx) %>, <%= var(ctx, :wx) %>;

      // (%cd2) input.offset = input + (tid.x * sx + tid.y * sy * ninput.x) * float_size
      mul.wide.u32  %vd0, %c3, <%= var(ctx, :sy) %>;
      mul.lo.u64    %vd0, %vd0, %cd4;
      mad.wide.u32  %vd0, %c2, <%= var(ctx, :sx) %>, %vd0;
      mad.lo.u64    %cd2, %vd0, <%= var(ctx, :float_size) %>, %cd0;
      <%= if offset(ctx, :input) > 0 do %>
        add.u64       %cd2, %cd2, <%= offset(ctx, :input) %>;
      <% end %>

      // (%cd1) w.offset = w + (tid.z * wx * wy * float_size)
      cvt.u64.u32   %vd0, %c4;
      mad.lo.u64    %cd1, %vd0, <%= var(ctx, :wx) * var(ctx, :wy) * var(ctx, :float_size) %>, %cd1;

      // (%cd3) output.offset = output + (tid.z * ntid.x * ntid.y + tid.y * ntid.x + tid.x) * float_size
      cvt.u64.u32   %vd1, %c2;
      mad.wide.u32  %vd0, %c3, %c5, %vd1;
      mul.wide.u32  %vd1, %c5, %c6;
      cvt.u64.u32   %vd2, %c4;
      mad.lo.u64    %vd0, %vd1, %vd2, %vd0;
      mad.lo.u64    %cd3, %vd0, <%= var(ctx, :float_size) %>, %cd0;
      <%= if offset(ctx, :output) > 0 do %>
        add.u64       %cd3, %cd3, <%= offset(ctx, :output) %>;
      <% end %>

      // (%vd0) dx = (ninput.x - wx) * float_size
      sub.u64       %vd0, %cd4, <%= var(ctx, :wx) %>;
      mul.lo.u64    %vd0, %vd0, <%= var(ctx, :float_size) %>;

      // %f0 - accumulator
      mov.f32       %f0, 0.0;
      // %c1 - wy
      mov.u32       %c1, <%= var(ctx, :wy) %>;

    loop_y:
      mov.u32       %v0, <%= var(ctx, :wx) %>;
    loop_x:
      // acc = acc + [input] * [w]
      ld.global.<%= var(ctx, :f) %> %f1, [%cd1];
      ld.global.<%= var(ctx, :f) %> %f2, [%cd2];
      mad.rn.<%= var(ctx, :f) %>    %f0, %f1, %f2, %f0;
      // next point
      add.u64       %cd1, %cd1, <%= var(ctx, :float_size) %>;
      add.u64       %cd2, %cd2, <%= var(ctx, :float_size) %>;
      // count x
      sub.u32       %v0, %v0, 1;
      setp.ne.u32   p, %v0, 0;
      @p bra        loop_x;
      // next line
      add.u64       %cd2, %cd2, %vd0;
      // count y
      sub.u32       %c1, %c1, 1;
      setp.ne.u32   p, %c1, 0;
      @p bra        loop_y;

      <%= include ctx, var(ctx, :activation), in: "f0", pred: "p" %>
      st.global.<%= var(ctx, :f) %> [%cd3], %f0;
      ret;
    <% end %>
    """
  end

  def vars(opts) do
    {x, y, z}    = opts |> Keyword.get(:size) |> Base.triple_size()
    {wx, wy, wz} = opts |> Keyword.get(:kernel_size) |> Base.triple_size()
    {sx, sy}     = opts |> Keyword.get(:stride) |> Base.stride()
    float_size   = opts |> Keyword.get(:float_size) |> Base.float_size()
    activation   = opts |> Keyword.get(:activation, :relu) |> Base.activation()
    f = "f#{float_size * 8}"

    ox = round((x - wx + sx) / sx)
    oy = round((y - wy + sy) / sy)
    oz = wz

    %{x: x, y: y, z: z,
      ox: ox, oy: oy, oz: oz,
      wx: wx, wy: wy, wz: wz,
      sx: sx, sy: sy,
      activation: activation,
      f: f, float_size: float_size}
  end
end
