defmodule Neuro.Nodes.Padding do
  alias Neuro.Nodes.Base
  use Base

  def __ptx__(_node) do
    """
    .version 5.0
    .target sm_30
    .address_size 64

    <%= defkernel ctx, "padding" do %>
      .reg .u64   %cd<3>;
      .reg .u32   %c<2>;
      <%= if var(ctx, :py_size) > 0 do %>
        .reg .<%= var(ctx, :f) %> %f<1>;
      <% end %>
      .reg .pred  p;

      ld.param.u64  %cd0, [pins];
      mov.u32       %c0, %tid.x;
      cvt.u64.u32   %cd1, %c0;

      // (%cd1) input.offset  = input  + tid.x * x * y * float_size
      // (%cd2) output.offset = output + tid.x * ox * oy * float_size
      mad.lo.u64    %cd2, %cd0, %cd1, <%= var(ctx, :ox) * var(ctx, :oy) * var(ctx, :float_size) %>;
      mad.lo.u64    %cd1, %cd0, %cd1, <%= var(ctx, :x) * var(ctx, :y) * var(ctx, :float_size) %>;

      <%= if var(ctx, :py_size) > 0 do %>
        mov.u32 %c0, <%= var(ctx, :py_size) %>;
      loop1:
        st.global.<%= var(ctx, :f) %> [%cd2], <%= var(ctx, :padding) %>;
        add.u64     %cd2, %cd2, <%= var(ctx, :float_size) %>;
        sub.u32     %c0, %c0, 1;
        setp.ne.u32 p, %c0, 0;
        @p bra      loop1;
      <% end %>

      <%= if var(ctx, :px) > 0 do %>
        mov.u32 %c0, <%= var(ctx, :px) %>;
      loop2:
        st.global.<%= var(ctx, :f) %> [%cd2], <%= var(ctx, :padding) %>;
        add.u64     %cd2, %cd2, <%= var(ctx, :float_size) %>;
        sub.u32     %c0, %c0, 1;
        setp.ne.u32 p, %c0, 0;
        @p bra      loop2;

        mov.u32 %c0, <%= var(ctx, :x) %>;
      loop3:
        ld.global.<%= var(ctx, :f) %> %f0, [%cd1];
        st.global.<%= var(ctx, :f) %> [%cd2], %f0;
        add.u64     %cd1, %cd1, <%= var(ctx, :float_size) %>;
        add.u64     %cd2, %cd2, <%= var(ctx, :float_size) %>;
        sub.u32     %c0, %c0, 1;
        setp.ne.u32 p, %c0, 0;
        @p bra      loop3;

        mov.u32 %c0, <%= var(ctx, :px) %>;
      loop4:
        st.global.<%= var(ctx, :f) %> [%cd2], <%= var(ctx, :padding) %>;
        add.u64     %cd2, %cd2, <%= var(ctx, :float_size) %>;
        sub.u32     %c0, %c0, 1;
        setp.ne.u32 p, %c0, 0;
        @p bra      loop4;
      <% else %>
        add.u64     %cd2, %cd2, <%= var(ctx, :x) * var(ctx, :y) * var(ctx, :float_size) %>;
      <% end %>

      <%= if var(ctx, :py_size) > 0 do %>
        mov.u32 %c1, <%= var(ctx, :py_size) %>;
      loop5:
        st.global.<%= var(ctx, :f) %> [%cd2], <%= var(ctx, :padding) %>;
        add.u64     %cd2, %cd2, <%= var(ctx, :float_size) %>;
        sub.u32     %c0, %c0, 1;
        setp.ne.u32 p, %c0, 0;
        @p bra      loop5;
      <% end %>
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
