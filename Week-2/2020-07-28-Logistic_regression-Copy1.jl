using Random
using Plots
using HDF5
using Statistics
using Base.Iterators

function load_dataset(train_file::String, test_file::String)
    
    X_train = convert(Array{Float64, 4}, h5read(train_file, "train_set_x"))
    y_train = convert(Array{Float64, 1}, h5read(train_file, "train_set_y"))
    
    X_test = convert(Array{Float64, 4}, h5read(test_file, "test_set_x"))
    y_test = convert(Array{Float64, 1}, h5read(test_file, "test_set_y"))
    
    num_features_train_X = size(X_train, 1) * size(X_train, 2) * size(X_train, 2)
    num_features_test_X = size(X_test, 1) * size(X_test, 2) * size(X_test, 2)
    
    X_train = reshape(X_train, (num_features_train_X, size(X_train, 4)))
    y_train = reshape(y_train, (1, size(y_train, 1)))
    
    X_test = reshape(X_test, (num_features_test_X, size(X_test, 4)))
    y_test = reshape(y_test, (1, size(y_test, 1)))
    
    X_train, y_train, X_test, y_test
    
end

X_train, y_train, X_test, y_test = load_dataset("train_catvnoncat.h5", "test_catvnoncat.h5");

@time size(X_train), size(y_train), size(X_test), size(y_test)

X_train, X_test= X_train/255, X_test/255;

@time size(X_train), size(y_train), size(X_test), size(y_test)

function σ(z) 
    """
    Compute the sigmoid of z
    """
    return one(z) / (one(z) + exp(-z))
end

function initialize(dim)
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = zeros(dim, 1)
    b = 2
    
    @assert(size(w) == (dim, 1))
    @assert(isa(b, Float64) || isa(b, Int64))
    
    return w, b
end

function propagate(w, b, X, Y)
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation
    """
    m = size(X, 2)
    
    # Forward prop
    Z = w'X .+ b
    ŷ = σ.(Z)
    
    @assert(size(ŷ) == size(Y))
    
    # Compute cost
    𝒥 = -1 * sum(Y .* log.(ŷ) .+ (1 .- Y) .* log.(1 .- ŷ))
    𝒥 /= m
    
    @assert(size(𝒥) == ())
    
    # Back-prop
    𝜕𝑧 = ŷ - Y
    @assert(size(𝜕𝑧) == size(ŷ) && size(𝜕𝑧) == size(Y))
        
    𝜕𝑤 = (1/m) * X * 𝜕𝑧'
    𝜕𝑏 = (1/m) * sum(𝜕𝑧)
    
    𝒥, Dict("𝜕𝑤" => 𝜕𝑤, "𝜕𝑏" => 𝜕𝑏)
end

function optimize(w, b, X, Y, num_iterations, 𝛼, print_cost)
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b
    """
    
    costs = Array{Float64, 2}(undef, num_iterations, 1)
    
    for i=1:num_iterations
        
        𝒥, 𝛻 = propagate(w, b, X, Y)
        
        𝜕𝑤, 𝜕𝑏 = 𝛻["𝜕𝑤"], 𝛻["𝜕𝑏"] 
        
        global 𝜕𝑤, 𝜕𝑏
        
        w -= 𝛼 .* 𝜕𝑤
        b -= 𝛼 .* 𝜕𝑏
        
        costs[i] = 𝒥
        
        if print_cost && i % 100 == 0
            println("Cost after iteration $i = $𝒥")
        end
    end
    
    params = Dict("w" => w, "b" => b)
    grads = Dict("𝜕𝑤" => 𝜕𝑤, "𝜕𝑏" => 𝜕𝑏)
    
    params, grads, costs
    
end

function predict(w, b, X)
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    m = size(X, 2)
    preds = zeros(1, m)
    
    ŷ = σ.(w'X .+ b)
    
    preds = [p > 0.5 ? 1 : 0 for p in Iterators.flatten(ŷ)]
    
    preds = reshape(preds, (1, m))
            
    @assert(size(preds) == (1, m))
    
    preds
end

function model(X_train, y_train, X_test, y_test, num_iterations, 𝛼, print_cost)
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # Initialize parameters
    w, b = initialize(size(X_train, 1))
    
    # Gradient descent
    params, grads, costs = optimize(w, b, X_train, y_train, num_iterations, 𝛼, print_cost)
    
    w, b = params["w"], params["b"]
    
    preds_test = predict(w, b, X_test)
    preds_train = predict(w, b, X_train)
    
    train_acc = 100 - mean(abs.(preds_train - y_train)) * 100
    test_acc = 100 - mean(abs.(preds_test - y_test)) * 100
    
    @show train_acc
    @show test_acc
    
    d = Dict(
        "costs" => costs, 
        "test_preds" => preds_test, 
        "train_preds" => preds_train,
        "w" => w,
        "b" => b,
        "𝛼" => 𝛼,
        "num_iterations" =>  num_iterations
    )
    
    d;
end

d = model(X_train, y_train, X_test, y_test, 2000, 0.005, true);

x = 1:2000;
y = d["costs"];

gr() # backend

plot(x, y, title = "Learning rate = 0.005", label="negative log-likelihood")
xlabel!("iteration")
ylabel!("cost")
