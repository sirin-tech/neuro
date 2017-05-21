defmodule Neuro.Nodes.Padding do
  alias Neuro.Nodes.Base
  alias Neuro.Nodes.Convolution
  use Base

  def __batch__(%{assigns: %{back_propagation: true, vars: vars}}) do
    [{:run, {"back", vars.block, vars.grid, []}}]
  end
  def __batch__(%{assigns: %{vars: vars}}) do
    [{:run, {"inference", vars.block, vars.grid, []}}]
  end

  def __ptx__(%{assings: %{back_propagation: true}}) do
    back_ptx()
  end
  def __ptx__(_node) do
    inference_ptx()
  end

  defp inference_ptx() do
    """
    .version 5.0
    .target sm_30
    .address_size 64

    <%= defkernel ctx, "inference" do %>
      .reg .u64   %cd<3>;
      .reg .u64   %tidx, %tidz;
      .reg .u32   %tidy;
      .reg .<%= var(ctx, :f) %> %f;
      .reg .pred  p;

      cvt.u64.u32   %tidx, %ctaid.z;
      mov.u32       %tidy, %ctaid.y;
      cvt.u64.u32   %tidz, %ctaid.x;

      // padding if x < px || x >= px + ix || y < py || y >= py + iy
      setp.lo.u64 p, %tidx, <%= var(ctx, :px) %>;
      setp.hs.or.u64 p, %tidx, <%= var(ctx, :px) + var(ctx, :x) %>, p;
      setp.lo.or.u32 p, %tidy, <%= var(ctx, :py) %>, p;
      setp.hs.or.u32 p, %tidy, <%= var(ctx, :py) + var(ctx, :y) %>, p;

      ld.param.u64  %cd0, [pins];

      // (%cd2) output.offset = output + (tid.x + tid.y * ox + tid.z * ox * oy) * float_size
      mad.wide.u32  %cd2, %tidy, <%= var(ctx, :ox) %>, %tidx;
      mad.lo.u64    %cd2, %tidz, <%= var(ctx, :ox) * var(ctx, :oy) %>, %cd2;
      mad.lo.u64    %cd2, %cd2, <%= var(ctx, :float_size) %>, %cd0;
      <%= if pin_offset(ctx, :output) > 0 do %>
        add.u64     %cd2, %cd2, <%= pin_offset(ctx, :output) %>;
      <% end %>

      @p bra padding;
      // (%cd1) input.offset  = input + (tid.x - px + (tid.y - py) * x + tid.z * x * y) * float_size
      sub.u64       %tidx, %tidx, <%= var(ctx, :px) %>;
      sub.u32       %tidy, %tidy, <%= var(ctx, :py) %>;
      mad.wide.u32  %cd1, %tidy, <%= var(ctx, :x) %>, %tidx;
      mad.lo.u64    %cd1, %tidz, <%= var(ctx, :x) * var(ctx, :y) %>, %cd1;
      mad.lo.u64    %cd1, %cd1, <%= var(ctx, :float_size) %>, %cd0;
      <%= if pin_offset(ctx, :input) > 0 do %>
        add.u64     %cd1, %cd1, <%= pin_offset(ctx, :input) %>;
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

  defp back_ptx() do
    """
    .version 5.0
    .target sm_30
    .address_size 64

    <%= defkernel ctx, "back", shared: u64.ptr do %>
      ret;
    <% end %>
    """
  end

  def vars(opts, %{gpu_info: info}) do
    {x, y, z} =  opts |> Keyword.get(:size) |> Base.triple_size()
    {px, py} =   opts |> Keyword.get(:padding_size) |> get_padding_size()
    padding =    opts |> Keyword.get(:padding, 0.0)
    float_size = opts |> Keyword.get(:float_size) |> Base.float_size()
    f = "f#{float_size * 8}"

    ox = x + px * 2
    oy = y + py * 2
    oz = z

    {block, grid} = Convolution.cta(ox, oy, oz, info)

    %{x: x, y: y, z: z,
      ox: ox, oy: oy, oz: oz,
      px: px, py: py, padding: padding,
      grid: grid, block: block,
      f: f, float_size: float_size}
  end

  defp get_padding_size({_, _} = tuple), do: tuple
  defp get_padding_size(x), do: {x, x}
end
