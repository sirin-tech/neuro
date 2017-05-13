defmodule Neuro.Nodes.Padding do
  alias Neuro.Nodes.Base
  use Base

  def __batch__(%{assigns: %{vars: vars}}) do
    [{"padding", {vars.ox, vars.oy, 1}, {1, 1, 1}, []}]
  end

  def __ptx__(_node) do
    """
    .version 5.0
    .target sm_30
    .address_size 64

    <%= defkernel ctx, "padding" do %>
      .reg .u64   %cd<3>;
      .reg .u64   %tidx;
      .reg .u32   %tidy;
      .reg .<%= var(ctx, :f) %> %f;
      .reg .pred  p;

      cvt.u64.u32   %tidx, %tid.x;
      mov.u32       %tidy, %tid.y;

      // padding if x < px || x >= px + ix || y < py || y >= py + iy
      setp.lo.u64 p, %tidx, <%= var(ctx, :px) %>;
      setp.hs.or.u64 p, %tidx, <%= var(ctx, :px) + var(ctx, :x) %>, p;
      setp.lo.or.u32 p, %tidy, <%= var(ctx, :py) %>, p;
      setp.hs.or.u32 p, %tidy, <%= var(ctx, :py) + var(ctx, :y) %>, p;

      ld.param.u64  %cd0, [pins];

      // (%cd2) output.offset = output + (tid.x + tid.y * ox) * float_size
      mad.wide.u32  %cd1, %tidy, <%= var(ctx, :ox) %>, %tidx;
      mad.lo.u64    %cd2, %cd1, <%= var(ctx, :float_size) %>, %cd0;
      <%= if offset(ctx, :output) > 0 do %>
        add.u64     %cd2, %cd2, <%= offset(ctx, :output) %>;
      <% end %>

      @p bra padding;
      // (%cd1) input.offset  = input + (tid.x - px + (tid.y - py) * x) * float_size
      sub.u64 %tidx, %tidx, <%= var(ctx, :px) %>;
      sub.u32 %tidy, %tidy, <%= var(ctx, :py) %>;
      mad.wide.u32 %cd1, %tidy, <%= var(ctx, :x) %>, %tidx;
      mad.lo.u64 %cd1, %cd1, <%= var(ctx, :float_size) %>, %cd0;
      <%= if offset(ctx, :input) > 0 do %>
        add.u64     %cd1, %cd1, <%= offset(ctx, :input) %>;
      <% end %>
      ld.global.<%= var(ctx, :f) %> %f, [%cd1];
      st.global.<%= var(ctx, :f) %> [%cd2], %f;
      ret;

    padding:
      st.global.<%= var(ctx, :f) %> [%cd2], <%= var(ctx, :padding) %>;
      ret;
    <% end %>
    """
  end

  def vars(opts) do
    {x, y, z} =  opts |> Map.get(:size) |> Base.triple_size()
    {px, py} =   opts |> Map.get(:padding_size) |> get_padding_size()
    padding =    opts |> Map.get(:padding, 0.0)
    float_size = opts |> Map.get(:float_size) |> Base.float_size()
    f = "f#{float_size * 8}"

    ox = x + px * 2
    oy = y + py * 2
    oz = z

    py_size = py * ox

    %{x: x, y: y, z: z,
      ox: ox, oy: oy, oz: oz,
      px: px, py: py, padding: padding,
      f: f, py_size: py_size,
      float_size: float_size}
  end

  defp get_padding_size({_, _} = tuple), do: tuple
  defp get_padding_size(x), do: {x, x}
end
