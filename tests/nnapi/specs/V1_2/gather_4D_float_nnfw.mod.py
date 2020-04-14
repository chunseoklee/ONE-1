# This test case is for 4d input gather operator.
# The input shape is [1,2,3,4] and this test produces output
# by referencing the data on axis 0 from the input to the indices value [0,0].
# In this case, the output shape changeds to [2,2,3,4] because it uses [0,0] indices on the 0 axis,
# and the data is configured as filling the 0th data of the input axis 0 twice.

# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1,2,3,4}") # a vector of 24 float32s
i2 = Input("op2", "TENSOR_INT32", "{2}") # another vector of 2 int32s
axis = Int32Scalar("axis", 0)
i3 = Output("op3", "TENSOR_FLOAT32", "{2,2,3,4}") # a vector of 48 float32s
model = model.Operation("GATHER", i1, axis, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.123456789123456789, 2.123456789123456789, 3.123456789123456789, 4.123456789123456789,
          5.123456789123456789, 6.123456789123456789, 7.123456789123456789, 8.123456789123456789,
          9.123456789123456789, 10.123456789123456789, 11.123456789123456789, 12.123456789123456789,
          13.123456789123456789, 14.123456789123456789, 15.123456789123456789, 16.123456789123456789,
          17.123456789123456789, 18.123456789123456789, 19.123456789123456789, 20.123456789123456789,
          21.123456789123456789, 22.123456789123456789, 23.123456789123456789, 24.123456789123456789],
          i2: # input 1
          [0, 0]}

output0 = {i3: # output 0
          [1.123456789123456789, 2.123456789123456789, 3.123456789123456789, 4.123456789123456789,
          5.123456789123456789, 6.123456789123456789, 7.123456789123456789, 8.123456789123456789,
          9.123456789123456789, 10.123456789123456789, 11.123456789123456789, 12.123456789123456789,
          13.123456789123456789, 14.123456789123456789, 15.123456789123456789, 16.123456789123456789,
          17.123456789123456789, 18.123456789123456789, 19.123456789123456789, 20.123456789123456789,
          21.123456789123456789, 22.123456789123456789, 23.123456789123456789, 24.123456789123456789,
          1.123456789123456789, 2.123456789123456789, 3.123456789123456789, 4.123456789123456789,
          5.123456789123456789, 6.123456789123456789, 7.123456789123456789, 8.123456789123456789,
          9.123456789123456789, 10.123456789123456789, 11.123456789123456789, 12.123456789123456789,
          13.123456789123456789, 14.123456789123456789, 15.123456789123456789, 16.123456789123456789,
          17.123456789123456789, 18.123456789123456789, 19.123456789123456789, 20.123456789123456789,
          21.123456789123456789, 22.123456789123456789, 23.123456789123456789, 24.123456789123456789]}

# Instantiate an example
Example((input0, output0))
