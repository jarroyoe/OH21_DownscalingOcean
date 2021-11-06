using DelimitedFiles, BSON
using Flux, Statistics
using Base.Iterators: repeated, partition

#Load CSVs
@info("Loading data set")
train_lr = [readdlm("./data/January/anobig_temp"*string(i)*".txt",',') for i in 1:42]
test_lr = [readdlm("./data/January/anobig_temp"*string(i)*".txt",',') for i in 43:60]

train_hr = [readdlm("./data/January/anohigh_temp"*string(i)*".txt",',') for i in 1:42]
test_hr = [readdlm("./data/January/anohigh_temp"*string(i)*".txt",',') for i in 43:60]

lr_size_x = 33
lr_size_y = 33
hr_size = 130

#Bundle training data into batches
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    Y_batch = Array{Float32}(undef, size(Y[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
        Y_batch[:, :, :, i] = Float32.(Y[idxs[i]])
    end
    return (X_batch, Y_batch)
end
batch_size = 10
mb_idxs = partition(1:length(train_lr), batch_size)
train_set = [make_minibatch(train_lr, train_hr, i) for i in mb_idxs]

# Prepare test set as one giant minibatch:
test_set = make_minibatch(test_lr, test_hr, 1:length(test_imgs))

#Define our CNN model
@info("Constructing model...")
model = Chain(
    Conv((9,9),1=>64,pad=1,relu),
    x -> maxpool(x,2),
    Dropout(0.2),

    Conv((3,3),64=>32,pad=1,relu),
    x -> maxpool(x,2),
    Dropout(0.2),

    Conv((5,5),32=>1,pad=1,relu),
    x -> maxpool(x,2),
    Dropout(0.2),

    x -> reshape(x,:,size(x,4)),
    Dense(round(Int64,lr_size_x/8)*round(Int64,lr_size_y/8),hr_size)
)

# Load model and datasets onto GPU, if enabled
train_set = gpu.(train_set)
test_set = gpu.(test_set)
model = gpu(model)

# Make sure our model is nicely precompiled before starting our training loop
model(train_set[1][1])

function loss(x,y)
    # We augment `x` a little bit here, adding in random noise
    x_aug = x .+ 0.1f0*gpu(randn(eltype(x), size(x)))

    y_hat = model(x_aug)
    return mean((y_hat-y)^2)
end
accuracy(x,y) = mean((model(x) .- y)^2)

opt = ADAM(0.001)

@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in 1:100
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)

    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    
    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
        BSON.@save "mnist_conv.bson" model epoch_idx acc
        best_acc = acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 10
        @warn(" -> We're calling this converged.")
        break
    end
end