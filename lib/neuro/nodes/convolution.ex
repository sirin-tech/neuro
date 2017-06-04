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
      .reg .u64   %cd<4>, %x, %y, %z;
      .reg .u32   %c<2>;
      .reg .<%= var(:f) %> %f<3>;
      .reg .pred  p;

      ld.param.u64  %cd0, [pins];
      ld.param.u64  %cd1, [shared];
      cvt.u64.u32   %x, %ctaid.z;
      cvt.u64.u32   %y, %ctaid.y;
      cvt.u64.u32   %z, %ctaid.x;

      // (%cd2) input.offset = input + (tid.x * sx + tid.y * sy * x) * float_size
      mul.lo.u64    %cd2, %x, <%= var(:sx) %>;
      mad.lo.u64    %cd2, %y, <%= var(:sy) * var(:x) %>, %cd2;
      mad.lo.u64    %cd2, %cd2, <%= var(:float_size) %>, %cd0;
      <%= if pin_offset(:input) > 0 do %>
        add.u64       %cd2, %cd2, <%= pin_offset(:input) %>;
      <% end %>

      // (%cd1) w.offset = w + tid.z * wx * wy * float_size
      mad.lo.u64    %cd1, %z, <%= var(:wx) * var(:wy) * var(:float_size) %>, %cd1;
      <%= if shared_offset(:weights) > 0 do %>
        add.u64     %cd1, %cd1, <%= shared_offset(:weights) %>;
      <% end %>

      // (%cd3) output.offset = output + (tid.z * ox * oy + tid.y * ox + tid.x) * float_size
      mad.lo.u64    %cd3, %y, <%= var(:ox) %>, %x;
      mad.lo.u64    %cd3, %z, <%= var(:ox) * var(:oy) %>, %cd3;
      mad.lo.u64    %cd3, %cd3, <%= var(:float_size) %>, %cd0;
      <%= if pin_offset(:output) > 0 do %>
        add.u64     %cd3, %cd3, <%= pin_offset(:output) %>;
      <% end %>

      // %f0 - accumulator
      mov.f32       %f0, 0.0;
      // %c1 - wy
      mov.u32       %c1, <%= var(:wy) %>;

    loop_y:
      mov.u32       %c0, <%= var(:wx) %>;
    loop_x:
      // acc = acc + [input] * [w]
      ld.global.<%= var(:f) %> %f1, [%cd1];
      ld.global.<%= var(:f) %> %f2, [%cd2];
      mad.rn.<%= var(:f) %>    %f0, %f1, %f2, %f0;
      // next point
      add.u64       %cd1, %cd1, <%= var(:float_size) %>;
      add.u64       %cd2, %cd2, <%= var(:float_size) %>;
      // count x
      sub.u32       %c0, %c0, 1;
      setp.ne.u32   p, %c0, 0;
      @p bra        loop_x;
      // next line
      add.u64       %cd2, %cd2, <%= (var(:x) - var(:wx)) * var(:float_size) %>;
      // count y
      sub.u32       %c1, %c1, 1;
      setp.ne.u32   p, %c1, 0;
      @p bra        loop_y;

      <%= include ctx, var(ctx, :activation), in: "f0", pred: "p" %>
      st.global.<%= var(:f) %> [%cd3], %f0;
      //st.global.<%= var(:f) %> [%cd3], %f1;
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
    {x, y, z}    = opts |> Keyword.get(:size) |> Base.triple_size()
    {wx, wy, wz} = opts |> Keyword.get(:kernel_size) |> Base.triple_size()
    {sx, sy}     = opts |> Keyword.get(:stride) |> Base.stride()
    activation   = opts |> Keyword.get(:activation, :relu) |> Base.activation()

    ox = round((x - wx + sx) / sx)
    oy = round((y - wy + sy) / sy)
    oz = wz

    {block, grid} = cta(ox, oy, oz, info)

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
    %{shared: %{weights: %{key => {vars.f, vars.wx * vars.wy * vars.wz}},
                biases:  %{key => {vars.f, vars.wx * vars.wy * vars.wz}}}}
  end
end
