import functorch

# Check that we get only one warning for the line
functorch.vmap(tdmodule, (None, 0))(td, params)
