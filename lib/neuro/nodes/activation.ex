defmodule Neuro.Nodes.Activation.Relu do
  def __ptx__(:body, opts) do
    in_reg  = Keyword.get(opts, :in, "f0")
    out_reg = Keyword.get(opts, :out, in_reg)
    p_reg   = Keyword.get(opts, :pred, "p")
    f       = Keyword.get(opts, :f, "f32")
    relu = """
           // ReLU activation
           setp.lt.#{f} #{p_reg}, %#{in_reg}, 0.0;
           @#{p_reg} mov.#{f} %#{out_reg}, 0.0;
           """
    if in_reg != out_reg do
      relu <> "\n@!#{p_reg} mov.#{f} %#{out_reg}, %#{in_reg};"
    else
      relu
    end
  end

  def __ptx__(:back_propagation, opts) do
    in_reg  = Keyword.get(opts, :in, "f0")
    out_reg = Keyword.get(opts, :out, in_reg)
    p_reg   = Keyword.get(opts, :pred, "p")
    f       = Keyword.get(opts, :f, "f32")
    relu = """
           // ReLU activation derivation
           setp.gt.#{f} #{p_reg}, %#{in_reg}, 0.0;
           @#{p_reg} mov.#{f} %#{out_reg}, 1.0;
           """
    if in_reg != out_reg do
      relu <> "\n@!#{p_reg} mov.#{f} %#{out_reg}, 0.0"
    else
      relu
    end
  end
end
