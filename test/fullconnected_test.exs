defmodule FullConnectedTest do
  use ExUnit.Case

  test "sample layer" do
    i = [0.1, 0.2, 0.3, 0.4,
         0.5, 0.6, 0.7, 0.8]
    w = [1.0, 2.0, 3.0, 4.0,
         5.0, 6.0, 7.0, 8.0,

         1.5, 0.7, 6.4, 3.2,
         2.1, 1.1, 0.1, 9.0]

    o = Neuro.FullConnected.run({i, w, 8}, 2)
    o = Enum.map(o, & Float.round(&1, 1))

    # 4.4 = 0.1 * 1.0 + 0.2 * 2.0 + 0.5 * 3.0 + 0.6 * 4.0
    # 5.4 = 0.2 * 1.0 + 0.3 * 2.0 + 0.6 * 3.0 + 0.7 * 4.0
    # ...
    assert o == [20.4, 12.5]
  end
end
