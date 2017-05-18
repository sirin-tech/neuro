defmodule Neuro.Nodes.FullyConnected do
  alias Neuro.Nodes.Base
  use Base

  def __batch__(%{assigns: %{vars: vars}}) do
    [{"fully_connected", vars.block, vars.grid, [:shared]}]
  end

  def __ptx__(_node) do
    """
    .version 4.3
    .target sm_20
    .address_size 64

    <%= defkernel ctx, "fully_connected", shared: u64.ptr do %>
      .reg .u64   %cd<5>, %x, %last;
    	.reg .<%= var(ctx, :f) %> %f<4>;
    	.reg .pred	p;

      ld.param.u64 	%cd0, [pins];
    	ld.param.u64 	%cd3, [shared];

      cvt.u64.u32   %x, %tid.x;
      <%= if :by in var(ctx, :cta_shape) do %>
        cvt.u64.u32 %cd4, %tid.y;
        mad.lo.u64  %x, %cd4, <%= var(ctx, :max_x) %>, %x;
      <% end %>
      <%= if :bz in var(ctx, :cta_shape) do %>
        cvt.u64.u32 %cd5, %tid.z;
        mad.lo.u64  %x, %cd4, <%= var(ctx, :max_x) * var(ctx, :max_y) %>, %x;
      <% end %>
      <%= if :gx in var(ctx, :cta_shape) do %>
        cvt.u64.u32 %cd4, %ctaid.x;
        mad.lo.u64  %x, %cd4, <%= var(ctx, :max_x) * var(ctx, :max_y) * var(ctx, :max_z) %>, %x;
      <% end %>

      // (%cd1) weight.offset = weight + tid.x * ix * float_size
      mad.lo.u64   %cd1, %x, <%= var(ctx, :x) * var(ctx, :float_size) %>, %cd3;
      <%= if shared_offset(ctx, :weights) > 0 do %>
        add.u64    %cd1, %cd1, <%= shared_offset(ctx, :weights) %>;
      <% end %>

      // (%cd3) biases.offset = biases + tid.x * float_size
      mad.lo.u64   %cd3, %x, <%= var(ctx, :float_size) %>, %cd3;
      <%= if shared_offset(ctx, :biases) > 0 do %>
        add.u64    %cd3, %cd3, <%= shared_offset(ctx, :biases) %>;
      <% end %>
      ld.global.<%= var(ctx, :f) %> %f3, [%cd3];

      // (%cd2) output.offset = output + tid.x * float_size + output.offset
      mad.lo.u64   %cd2, %x, <%= var(ctx, :float_size) %>, %cd0;
      <%= if offset(ctx, :output) > 0 do %>
        add.u64    %cd2, %cd2, <%= offset(ctx, :output) %>;
      <% end %>

      // (%cd0) input.offset = pins + input.offset
      <%= if offset(ctx, :input) > 0 do %>
        add.u64    %cd0, %cd0, <%= offset(ctx, :input) %>;
      <% end %>

      // %f0 - accumulator
      mov.<%= var(ctx, :f) %>       %f0, 0.0;
      add.u64                       %last, %cd0, <%= var(ctx, :x) * var(ctx, :float_size) %>;
    loop_x:
      ld.global.<%= var(ctx, :f) %> %f1, [%cd0];
      ld.global.<%= var(ctx, :f) %> %f2, [%cd1];
      // accumulator = accumulator + w[x] * input[x]
      mad.rn.<%= var(ctx, :f) %>    %f0, %f1, %f2, %f0;
      // next values
      add.u64                       %cd0, %cd0, <%= var(ctx, :float_size) %>;
      add.u64                       %cd1, %cd1, <%= var(ctx, :float_size) %>;
      // count x
      setp.lo.u64                   p, %cd0, %last;
      @p bra                        loop_x;

      // bias
      add.f32 %f0, %f0, %f3;

      // relu activation
      <%= include ctx, var(ctx, :activation), in: "f0", pred: "p" %>
      st.global.<%= var(ctx, :f) %> [%cd2], %f0;
      ret;
    <% end %>
    """
  end

  def vars(opts, %{gpu_info: info}) do
    x              = opts |> Keyword.get(:size) |> Base.plane_size()
    ox             = opts |> Keyword.get(:out_size) |> Base.plane_size()
    float_size     = opts |> Keyword.get(:float_size) |> Base.float_size()
    activation     = opts |> Keyword.get(:activation, :relu) |> Base.activation()
    f = "f#{float_size * 8}"

    {max_x, max_y, max_z} = info[:max_block]
    {block, grid, cta_shape} = cond do
      ox <= max_x ->
        {{ox, 1, 1}, {1, 1, 1}, ~w(bx)a}
      ox <= max_x * max_y ->
        by = div(ox, max_x) + (if rem(ox, max_x) > 0, do: 1, else: 0)
        {{max_x, by, 1}, {1, 1, 1}, ~w(bx by)a}
      ox <= max_x * max_y * max_z ->
        b = max_x * max_y
        bz = div(ox, b) + (if rem(ox, b) > 0, do: 1, else: 0)
        {{max_x, max_y, bz}, {1, 1, 1}, ~w(bx by bz)a}
      true ->
        b = max_x * max_y * max_z
        gx = div(ox, b) + (if rem(ox, b) > 0, do: 1, else: 0)
        {{max_x, max_y, max_z}, {gx, 1, 1}, ~w(bx by bz gx)a}
    end

    %{x: x, y: 1, z: 1,
      ox: ox, oy: 1, oz: 1,
      neurons: ox, weights: x * ox,
      activation: activation,
      block: block, grid: grid, cta_shape: cta_shape,
      max_x: max_x, max_y: max_y, max_z: max_z,
      f: f, float_size: float_size}
  end
end
