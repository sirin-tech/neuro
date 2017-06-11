defmodule Neuro.Nodes.Error do
  use Cuda.Graph.GPUNode

  def __assigns__(_id, opts, _env) do
    size = Neuro.Nodes.Base.plane_size(Keyword.fetch!(opts, :size))
    %{vars: %{size: size}}
  end

  def __pins__(%{vars: %{size: size}, env: env}) do
    f = Cuda.Env.f(env)
    type = {f, size}
    [input(:input, type, :activation),
     input(:reply, type),
     output(:output, type),
     output(:error, f)]
  end

  def __batch__(_) do
    [{:run, {"error", {1, 1, 1}, {1, 1, 1}}}]
  end

  def __ptx__(_) do
    """
    <%= defkernel ctx, "error" do %>
      .reg .u64 %input, %reply, %output, %error, %in, %count;
      .reg .<%= var(:f) %> %x, %y, %loss;
      .reg .pred p;

      ld.param.u64  %input, [pins];

      <%= if pin_offset(:reply) > 0 do %>
        add.u64     %reply, %input, <%= pin_offset(:reply) %>;
      <% else %>
        mov.u64     %reply, %input;
      <% end %>

      <%= if pin_offset(:error) > 0 do %>
        add.u64     %error, %input, <%= pin_offset(:error) %>;
      <% else %>
        mov.u64     %error, %input;
      <% end %>

      <%= if pin_offset(:output) > 0 do %>
        add.u64     %output, %input, <%= pin_offset(:output) %>;
      <% else %>
        mov.u64     %output, %input;
      <% end %>

      <%= if pin_offset(:input) > 0 do %>
        add.u64     %input, %input, <%= pin_offset(:input) %>;
      <% end %>

      mov.<%= var(:f) %>        %loss, 0.0;

      mov.u64                   %count, 0;
      mov.u64                   %in, %input;
    loss_loop:
      ld.global.<%= var(:f) %>  %x, [%in];
      ld.global.<%= var(:f) %>  %y, [%reply];
      sub.<%= var(:f) %>        %x, %x, %y;
      mad.rn.<%= var(:f) %>     %loss, %x, %x, %loss;
      add.u64                   %in, %in, <%= var(:float_size) %>;
      add.u64                   %reply, %reply, <%= var(:float_size) %>;
      add.u64                   %count, %count, 1;
      setp.lo.u64               p, %count, <%= var(:size) %>;
      @p bra loss_loop;

      mul.rn.<%= var(:f) %>     %loss, %loss, 0.5;
      st.global.<%= var(:f) %>  [%error], %loss;

      mov.u64                   %count, 0;
    loop:
      ld.global.<%= var(:f) %>  %x, [%input];
      sub.<%= var(:f) %>        %x, %x, %loss;
      st.global.<%= var(:f) %>  [%output], %x;
      add.u64                   %input, %input, <%= var(:float_size) %>;
      add.u64                   %output, %output, <%= var(:float_size) %>;
      add.u64                   %count, %count, 1;
      setp.lo.u64               p, %count, <%= var(:size) %>;
      @p bra loop;

      ret;
    <% end %>
    """
  end
end
