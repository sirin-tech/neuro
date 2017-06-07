defmodule Neuro.Mixfile do
  use Mix.Project

  def project do
    [app: :neuro,
     version: "0.1.0",
     elixir: "~> 1.4",
     elixirc_paths: paths(),
     build_embedded: Mix.env == :prod,
     start_permanent: Mix.env == :prod,
     deps: deps()]
  end

  def application do
    # Specify extra applications you'll use from Erlang/Elixir
    [extra_applications: [:logger]]
  end

  defp deps do
    [{:cuda, path: "../cuda"},
     {:gen_stage,  "~> 0.12.0"}]
  end

  defp paths do
    ["lib", Path.join(~w(test support))]
  end
end
