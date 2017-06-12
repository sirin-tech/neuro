defmodule Neuro.Nodes.Convolution do
  alias Neuro.Nodes.Base
  use Base

  def __batch__(%{assigns: %{back_propagation: true, vars: vars}}) do
    [{:run, {"back", vars.block, vars.grid, [:shared]}},
     {:run, {"delta_w", {1, 1, 1}, {vars.wz, vars.wy, vars.wx}, [:shared]}},
     {:run, {"delta_b", {1, 1, 1}, {vars.wz, 1, 1}, [:shared]}}]
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
      .reg .u64   %x, %y, %z, %pins, %i_ptr, %o_ptr, %w_ptr, %b_ptr, %cx, %cy,
                  %py, %px1, %px2, %sx, %sy;
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

      mul.lo.u64    %sx, %x, <%= var(:sx) %>;
      mul.lo.u64    %sy, %y, <%= var(:sy) %>;

      // input.offset = input + (tid.x * sx + tid.y * sy * x) * float_size
      mad.lo.u64    %i_ptr, %sy, <%= var(:x) %>, %sx;
      mad.lo.u64    %i_ptr, %i_ptr, <%= var(:float_size) %>, %pins;
      <%= if pin_offset(:input) > 0 do %>
        add.u64     %i_ptr, %i_ptr, <%= pin_offset(:input) %>;
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

      // b.offset = b + tid.z * float_size
      mad.lo.u64    %b_ptr, %z, <%= var(:float_size) %>, %w_ptr;
      <%= if shared_offset(:biases) > 0 do %>
        add.u64     %b_ptr, %b_ptr, <%= shared_offset(:biases) %>;
      <% end %>

      // w.offset = w + tid.z * wx * wy * float_size
      mad.lo.u64    %w_ptr, %z, <%= var(:wx) * var(:wy) * var(:float_size) %>, %w_ptr;
      <%= if shared_offset(:weights) > 0 do %>
        add.u64     %w_ptr, %w_ptr, <%= shared_offset(:weights) %>;
      <% end %>

      // %acc - accumulator
      mov.f32       %acc, 0.0;
      mov.u64       %cy, <%= var(:wy) %>;

      // add Y padding before
      <%= if var(:padding) and var(:py) > 0 do %>
        setp.hs.u64     p, %sy, <%= var(:py) %>;
        @p bra          bottom_padding;
        sub.u64         %py, <%= var(:py) %>, %sy;
        sub.u64         %cy, <%= var(:wy) %>, %py;
        mad.lo.u64      %i_ptr, %py, <%= var(:x) * var(:float_size) %>, %i_ptr;
        <%= if var(:pv) == 0.0 do %>
          mad.lo.u64  %w_ptr, %py, <%= var(:wx) * var(:float_size) %>, %w_ptr;
        <% else %>
        loop_py1:
          ld.global.<%= var(:f) %> %w, [%w_ptr];
          mad.rn.<%= var(:f) %>    %acc, %w, <%= var(:pv) %>, %acc;
          add.u64                  %w_ptr, %w_ptr, <%= var(:float_size) %>;
          sub.u64                  %py, %py, 1;
          setp.ne.u64              p, %py, 0;
          @p bra                   loop_py1
        <% end %>
      bottom_padding:
        mov.u64         %py, 0;
        setp.hi.u64     p, %sy, <%= var(:y) - var(:wy) + var(:py) %>;
        @p sub.u64      %py, %sy, <%= var(:y) - var(:wy) + var(:py) %>;
        @p sub.u64      %cy, %cy, %py;
        @p sub.u64      %z, %sy, <%= var(:py) %>;
      <% end %>
      <%= if var(:padding) and var(:px) > 0 do %>
        mov.u64         %px1, 0;
        mov.u64         %px2, 0;
        setp.lo.u64     p, %sx, <%= var(:px) %>;
        @p sub.u64      %px1, <%= var(:px) %>, %sx;
        @p mad.lo.u64   %i_ptr, %px1, <%= var(:float_size) %>, %i_ptr;
        setp.hi.u64     p, %sx, <%= var(:x) - var(:wx) + var(:px) %>;
        @p sub.u64      %px2, %sx, <%= var(:x) - var(:wx) + var(:px) %>;
        @p sub.u64      %z, %sx, <%= var(:px) %>;
      <% end %>
      <%= if var(:padding) do %>
        sub.u64         %i_ptr, %i_ptr, <%= (var(:px) + var(:py) * var(:x)) * var(:float_size) %>;
      <% end %>

    loop_y:
      // add X padding before
      <%= if var(:padding) and var(:px) > 0 do %>
        sub.u64       %cx, <%= var(:wx) %>, %px1;
        sub.u64       %cx, %cx, %px2;
        setp.eq.u64   p, %px1, 0;
        @p bra        loop_x;
        <%= if var(:pv) == 0.0 do %>
          mad.lo.u64  %w_ptr, %px1, <%= var(:float_size) %>, %w_ptr;
        <% else %>
          mov.u64     %z, %px1;
        loop_px1:
          ld.global.<%= var(:f) %> %w, [%w_ptr];
          mad.rn.<%= var(:f) %>    %acc, %w, <%= var(:pv) %>, %acc;
          add.u64                  %w_ptr, %w_ptr, <%= var(:float_size) %>;
          sub.u64                  %z, %z, 1;
          setp.ne.u64              p, %z, 0;
          @p bra                   loop_px1
        <% end %>
      <% else %>
        mov.u64       %cx, <%= var(:wx) %>;
      <% end %>

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
      setp.hi.u64   p, %cx, 0;
      @p bra        loop_x;

      // add X padding after
      <%= if var(:padding) and var(:px) > 0 do %>
        setp.eq.u64   p, %px2, 0;
        @p bra        skip_px2;
        <%= if var(:pv) == 0.0 do %>
          mad.lo.u64  %w_ptr, %px2, <%= var(:float_size) %>, %w_ptr;
        <% else %>
          mov.u64     %z, %px2;
        loop_px2:
          ld.global.<%= var(:f) %> %w, [%w_ptr];
          mad.rn.<%= var(:f) %>    %acc, %w, <%= var(:pv) %>, %acc;
          add.u64                  %w_ptr, %w_ptr, <%= var(:float_size) %>;
          sub.u64                  %z, %z, 1;
          setp.ne.u64              p, %z, 0;
          @p bra                   loop_px2
        <% end %>
        skip_px2:
      <% end %>

      // next line
      add.u64       %i_ptr, %i_ptr, <%= (var(:x) - var(:wx)) * var(:float_size) %>;
      <%= if var(:padding) and var(:px) > 0 do %>
        mad.lo.u64  %i_ptr, %px1, <%= var(:float_size) %>, %i_ptr;
        mad.lo.u64  %i_ptr, %px2, <%= var(:float_size) %>, %i_ptr;
      <% end %>

      // count y
      sub.u64       %cy, %cy, 1;
      setp.hi.u64   p, %cy, 0;
      @p bra        loop_y;

      // add Y padding after
      <%= if var(:padding) and var(:py) > 0 do %>
        setp.eq.u64   p, %py, 0;
        @p bra        skip_py;
        <%= if var(:pv) != 0.0 do %>
        loop_py2:
          ld.global.<%= var(:f) %> %w, [%w_ptr];
          mad.rn.<%= var(:f) %>    %acc, %w, <%= var(:pv) %>, %acc;
          add.u64                  %w_ptr, %w_ptr, <%= var(:float_size) %>;
          sub.u64                  %py, %py, 1;
          setp.ne.u64              p, %py, 0;
          @p bra                   loop_py2
        <% end %>
        skip_py:
      <% end %>

      // add bias
      ld.global.<%= var(:f) %>  %w, [%b_ptr];
      add.<%= var(:f) %>        %acc, %acc, %w;

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
      <%= if var(:padding) do %>
        add.u64     %loss_ptr, %loss_ptr, <%= (var(:px) + var(:py) * var(:ox)) * var(:float_size) %>;
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
      <%= if var(:padding) and var(:py) > 0 do %>
        add.u64     %ly, %ly, <%= var(:py) %>;
      <% end %>

    y_loop:
      setp.hi.u64   p, %ly, <%= var(:oy) - 1 %>;
      @p bra        skip_y;
      mov.u64       %cx, 0;
      mov.u64       %lx, %x;
      <%= if var(:padding) and var(:px) > 0 do %>
        add.u64     %lx, %lx, <%= var(:px) %>;
      <% end %>

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
      <%= if var(:padding) and var(:py) > 0 do %>
        add.u64     %tmp, %tmp, <%= var(:py) %>;
      <% end %>
      mul.lo.u64    %tmp, %tmp, <%= var(:ox) * var(:float_size) %>;
      sub.u64       %loss_ptr, %loss_base, %tmp;
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

    <%= defkernel ctx, "delta_w", shared: u64.ptr do %>
      .reg .u64 %x, %y, %z, %dw_ptr, %inference_ptr, %loss_ptr, %loss_base,
                %cx, %cy, %cz;
      .reg .<%= var(:f) %> %dw, %inference, %loss;
      .reg .pred p;
      <%= if var(:padding) and var(:px) > 0 do %>
        .reg.u64  %px;
      <% end %>
      <%= if var(:padding) and var(:py) > 0 do %>
        .reg.u64  %py;
      <% end %>

      ld.param.u64  %inference_ptr, [pins];
      ld.param.u64  %dw_ptr, [shared];

      cvt.u64.u32   %x, %ctaid.z;
      cvt.u64.u32   %y, %ctaid.y;
      cvt.u64.u32   %z, %ctaid.x;

      // dw_ptr = (z * wx * wy + y * wx + x) * float_size
      mad.lo.u64    %cx, %y, <%= var(:wx) %>, %x;
      mad.lo.u64    %cx, %z, <%= var(:wx) * var(:wy) %>, %cx;
      mad.lo.u64    %dw_ptr, %cx, <%= var(:float_size) %>, %dw_ptr;
      <%= if shared_offset(:dw) > 0 do %>
        add.u64     %dw_ptr, %dw_ptr, <%= shared_offset(:dw) %>;
      <% end %>

      // loss
      <%= if pin_offset(:output) > 0 do %>
        add.u64     %loss_ptr, %inference_ptr, <%= pin_offset(:output) %>;
      <% else %>
        mov.u64     %loss_ptr, %inference_ptr;
      <% end %>

      // inference = (y * ix + x) * float_size
      mad.lo.u64    %cx, %y, <%= var(:x) %>, %x;
      mad.lo.u64    %inference_ptr, %cx, <%= var(:float_size) %>, %inference_ptr;
      <%= if pin_offset(:inference) > 0 do %>
        add.u64     %inference_ptr, %inference_ptr, <%= pin_offset(:inference) %>;
      <% end %>

      <%= if var(:padding) do %>
        <%= if var(:px) > 0 do %>
          mov.u64         %px, 0;
          setp.lo.u64     p, %x, <%= var(:px) %>;
          @p sub.u64      %cx, <%= var(:px) %>, %x;
          @p mad.lo.u64   %loss_ptr, %cx, <%= var(:float_size) %>, %loss_ptr;
          @p mov.u64      %px, %cx;
          setp.hi.u64     p, %x, <%= var(:wx) - 1 - var(:px) %>;
          @p sub.u64      %cx, %x, <%= var(:wx) - 1 - var(:px) %>;
          @p add.u64      %px, %px, %cx;
          @p mul.lo.u64   %cx, %cx, <%= var(:float_size) %>;
          @p sub.u64      %inference_ptr, %inference_ptr, %cx;
        <% end %>
        <%= if var(:py) > 0 do %>
          mov.u64         %py, 0;
          setp.lo.u64     p, %y, <%= var(:py) %>;
          @p sub.u64      %cx, <%= var(:py) %>, %y;
          @p mad.lo.u64   %loss_ptr, %cx, <%= var(:ox) * var(:float_size) %>, %loss_ptr;
          @p mov.u64      %py, %cx;
          setp.hi.u64     p, %y, <%= var(:wy) - 1 - var(:py) %>;
          @p sub.u64      %cx, %y, <%= var(:wy) - 1 - var(:py) %>;
          @p add.u64      %py, %py, %cx;
          @p mul.lo.u64   %cx, %cx, <%= var(:x) * var(:float_size) %>;
          @p sub.u64      %inference_ptr, %inference_ptr, %cx;
        <% end %>
      <% end %>

      mov.u64       %loss_base, %loss_ptr;
      mov.u64       %cz, 0;
      mov.<%= var(:f) %> %dw, 0.0;

    loop_z:
      mov.u64       %cy, 0;
      <%= if var(:padding) and var(:py) > 0 do %>
        add.u64     %cy, %cy, %py;
      <% end %>

    loop_y:
      mov.u64       %cx, 0;
      <%= if var(:padding) and var(:px) > 0 do %>
        add.u64     %cx, %cx, %px;
      <% end %>

    loop_x:
      ld.global.<%= var(:f) %>  %loss, [%loss_ptr];
      ld.global.<%= var(:f) %>  %inference, [%inference_ptr];
      mad.rn.<%= var(:f) %>     %dw, %loss, %inference, %dw;

      add.u64       %loss_ptr, %loss_ptr, <%= var(:float_size) %>;
      add.u64       %inference_ptr, %inference_ptr, <%= var(:float_size) %>;
      add.u64       %cx, %cx, 1;
      setp.lo.u64   p, %cx, <%= var(:ox) %>;
      @p bra        loop_x;

      <%= if var(:padding) and var(:px) > 0 do %>
        mad.lo.u64  %loss_ptr, %px, <%= var(:float_size) %>, %loss_ptr;
      <% end %>
      add.u64       %inference_ptr, %inference_ptr, <%= var(:float_size) * (var(:x) - var(:ox) + var(:px)) %>;
      add.u64       %cy, %cy, 1;
      setp.lo.u64   p, %cy, <%= var(:oy) %>;
      @p bra        loop_y;

      mov.u64       %loss_ptr, %loss_base;
      add.u64       %inference_ptr, %inference_ptr, <%= var(:float_size) * (var(:x) - var(:ox) + var(:px) + (var(:y) - var(:oy) + var(:py)) * var(:x)) %>;
      add.u64       %cz, %cz, 1;
      setp.lo.u64   p, %cz, <%= var(:z) %>;
      @p bra        loop_z;

      div.approx.<%= var(:f) %> %dw, %dw, <%= (var(:ox) - var(:px)) * (var(:oy) - var(:py)) * var(:z) %>.0;

      ld.global.<%= var(:f) %>  %loss, [%dw_ptr];
      add.<%= var(:f) %>        %loss, %loss, %dw;
      st.global.<%= var(:f) %>  [%dw_ptr], %loss;

      ret;
    <% end %>

    <%= defkernel ctx, "delta_b", shared: u64.ptr do %>
      .reg .u64 %db_ptr, %loss_ptr, %c, %z;
      .reg .<%= var(:f) %> %db, %loss;
      .reg .pred p;

      cvt.u64.u32   %z, %ctaid.x;
      ld.param.u64  %loss_ptr, [pins];
      ld.param.u64  %db_ptr, [shared];

      mad.lo.u64    %loss_ptr, %z, <%= var(:ox) * var(:oy) * var(:float_size) %>, %loss_ptr;
      <%= if pin_offset(:output) > 0 do %>
        add.u64     %loss_ptr, %loss_ptr, <%= pin_offset(:output) %>;
      <% end %>

      mad.lo.u64    %db_ptr, %z, <%= var(:float_size) %>, %db_ptr;
      <%= if shared_offset(:db) > 0 do %>
        add.u64     %db_ptr, %db_ptr, <%= shared_offset(:db) %>;
      <% end %>

      mov.u64                   %c, 0;
      mov.<%= var(:f) %>        %db, 0.0;
    loop:
      ld.global.<%= var(:f) %>  %loss, [%loss_ptr];
      add.<%= var(:f) %>        %db, %db, %loss;
      add.u64                   %loss_ptr, %loss_ptr, <%= var(:float_size) %>;
      add.u64                   %c, %c, 1;
      setp.lo.u64               p, %c, <%= var(:ox) * var(:oy) %>;
      @p bra                    loop;

      div.approx.<%= var(:f) %> %db, %db, <%= var(:ox) * var(:oy) %>.0;
      ld.global.<%= var(:f) %>  %loss, [%db_ptr];
      add.<%= var(:f) %>        %loss, %loss, %db;
      st.global.<%= var(:f) %>  [%db_ptr], %loss;

      ret;
    <% end %>
    """
  end

  def vars(opts, %{gpu_info: info}) do
    {x, y, z}    = opts |> Keyword.get(:size) |> Base.triple_size()
    {wx, wy, wz} = opts |> Keyword.get(:kernel_size) |> Base.triple_size()
    {sx, sy}     = opts |> Keyword.get(:stride) |> Base.stride()
    activation   = opts |> Keyword.get(:activation, :relu) |> Base.activation()
    padding      = opts |> Keyword.get(:padding, nil) |> padding

    {padding, px, py, pv} = case padding do
      false ->
        {false, 0, 0, 0.0}
      padding ->
        {px, py} = Keyword.get(padding, :padding_size, 1) |> padding_size()
        {true, px, py, Keyword.get(padding, :padding, 0.0)}
    end

    ox = round((x + px * 2 - wx + sx) / sx)
    oy = round((y + py * 2 - wy + sy) / sy)
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
      padding: padding, px: px, py: py, pv: pv,
      grid: grid, block: block,
      activation: activation}# |> IO.inspect
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
    w_size = vars.wx * vars.wy * vars.wz
    b_size = vars.wz
    shared = %{weights: %{key => {vars.f, w_size}},
               biases:  %{key => {vars.f, b_size}}}
    shared = if vars.training do
      Map.merge(shared, %{states: %{key => {vars.f, vars.ox * vars.oy * vars.oz}}})
    else
      shared
    end
    shared = if vars.back_propagation do
      Map.merge(shared, %{dw: %{key => {vars.f, w_size}},
                          db: %{key => {vars.f, b_size}}})
    else
      shared
    end
    %{shared: shared}
  end

  defp padding_size({_, _} = tuple), do: tuple
  defp padding_size(x) when is_integer(x), do: {x, x}
  defp padding_size(_), do: {1, 1}

  defp padding(nil), do: false
  defp padding([]), do: false
  defp padding(n) when is_integer(n), do: [padding_size: {n, n}]
  defp padding({_, _} = p), do: [padding_size: p]
  defp padding(p) when is_list(p) do
    if Keyword.keyword?(p), do: p, else: false
  end
  defp padding(_), do: false
end
