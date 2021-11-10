using DelimitedFiles, BSON, CUDA
using Flux, Statistics, StatsBase, Random
using Printf, Plots
using Base.Iterators: repeated, partition
CUDA.allowscalar(true)

#Load CSVs
@info("Loading data set")
random_split = shuffle(0:720)
train_size = 0.7
test_train = collect(partition(random_split,Int(ceil(length(random_split)*train_size))))

maxTemperature = 30

train_lr = [readdlm("./data/ANOBIG smaller_region/anobig_temp"*string(i)*".txt",',') for i in test_train[1]]/maxTemperature
test_lr = [readdlm("./data/ANOBIG smaller_region/anobig_temp"*string(i)*".txt",',') for i in test_train[2]]/maxTemperature

train_hr = [readdlm("./data/ANOHIGH smaller_region/anohigh_temp"*string(i)*".txt",',') for i in test_train[1]]/maxTemperature
test_hr = [readdlm("./data/ANOHIGH smaller_region/anohigh_temp"*string(i)*".txt",',') for i in test_train[2]]/maxTemperature

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
test_set = make_minibatch(test_lr, test_hr, 1:length(test_lr))

lr_size_x = size(train_set[1][1],1)
lr_size_y = size(train_set[1][1],2)
hr_size_x = size(train_set[1][2],1)
hr_size_y = size(train_set[1][2],2)

#Define our CNN model
@info("Constructing model...")
model = Chain(
    Conv((9,9),1=>64,pad=(1,1),relu),
    x -> maxpool(x,(1,1)),
    Dropout(0.4),

    Conv((3,3),64=>32,pad=(30,30),relu),
    x -> maxpool(x,(1,1)),
    Dropout(0.4),

    Conv((5,4),32=>1,pad=(0,4),sigmoid),
    x -> maxpool(x,(1,1)),
    Dropout(0.4),

    x -> reshape(x,:,size(x,4))
    #Dense(99,hr_size_x*hr_size_y)
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
    y = CuArray(reshape(y,(hr_size_x*hr_size_y,size(x,4))))
    return psnr(y_hat,y,1)
end
function accuracy(x,y)
    y = CuArray(reshape(y,(hr_size_x*hr_size_y,length(test_train[2]))))
    psnr(model(x),y,1)
end

opt = ADAM(0.001)

@info("Beginning training loop...")
best_acc = Inf
last_improvement = 0
@time for epoch_idx in 1:100
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)

    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    
    #If our accuracy is good enough, quit out.
    if acc <= 0.05
        @info(" -> Early-exiting: We reached our target accuracy of 95%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc <= best_acc
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

#first_predict = Array(reshape(model(test_set[1])[:,1],(hr_size_x,hr_size_y)))
#first_truth = Array(test_set[2][:,:,:,1])
#display(heatmap(first_truth))
#display(heatmap(first_predict))
