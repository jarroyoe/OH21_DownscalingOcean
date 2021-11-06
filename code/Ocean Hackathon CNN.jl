using NetCDF
using Flux

#Load CDF 
data = ncread("","")

#Temperature: 130x130x50x60
#Velocity: 130x129x50x60
lr_size_x = 130
lr_size_y = 130
hr_size = 130

#Bundle training data into batches
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

#Define our CNN model
model = Chain(
    Conv((9,9,9),1=>64,pad=1,relu),
    x -> maxpool(x,2),

    Conv((3,3,3),64=>32,pad=1,relu),
    x -> maxpool(x,2),

    Conv((5,5,5),32=>1,pad=1,relu),
    x -> maxpool(x,2),

    x -> reshape(x,:,size(x,5)),
    Dense(floor(Int64,lr_size_x/8)*floor(Int64,lr_size_y/8),hr_size)
)

