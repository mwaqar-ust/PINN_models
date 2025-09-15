clc; clearvars; clear;
% Dr. M. Waqar (Research Assistant Professor, HKKUST)
% ## NOTE 1 : This script is motivated by the blog-post of Ben Moseley. The link to his
% code on jupyter notebook can be found from the following link
% https://github.com/benmoseley/harmonic-oscillator-pinn/blob/main/Harmonic%20oscillator%20PINN.ipynb
% ## NOTE 2 : The spring system considered here is an under-damped system.
% ## NOTE 3 : This script requires MATLAB-2021a or above. 
% ## NOTE 4 : The weights initialization is seeded using rng(123).

%% CONTROL UNIT

global d w0 m k mu  

m  = 1; % mass of the object
d = 2; % damping coefficient
mu = 2*d; % damping coefficient
w0 = 20; % frequency
k = w0^2; % spring constant
T = 1; % Maximum time
method = 'ADAM'; %Choose one: 'ADAM', 'LBFGS'. >>ADAM gives better results!
NN_choice = 'PINN'; % Choose one: 'PINN', 'NN'


%%  ANALYTICAL SOLUTION
t = linspace(0,T,500); % time vector for analytical solution
x = oscillator(t); % analytical solution

figure(1); hold on; box on; plottools on;LW = 'LineWidth';
set(gcf, 'color', 'white');set(gca,'fontsize',16);
set(gca, 'FontName', 'Times New Roman');
xlabel('$t$','Interpreter','Latex');ylabel('$x(t)$','Interpreter','Latex')
plot(t,x,'k-',LW,2,'Display','Analytical solution')


%% DATA CURATION FOR TRAINING

% initial condition
t0IC = 0; 
x0IC = 1;

%traning points
sample_id = 20:20:floor(length(t)/2.5);
t_data = t(sample_id);
x_data = x(sample_id);

% initial condition and traning points
T0 = [t0IC t_data];
X0 = [x0IC x_data];


figure(1); 
plot(T0,X0,'o','MarkerSize',10,'Display','Training data',LW,2)
legend('Interpreter','latex')
disp('Initial settings completed.')

%% Training Inputs Datapoints

numInternalCollocationPoints = 2000; % random generation of input variable for training
pointSet = sobolset(1);
points = net(pointSet,numInternalCollocationPoints);

dataT = T*points(:,1);
ds = arrayDatastore(dataT);

%% NN Architecture
numLayers = 3; %(including input and output layer)
numNeurons = 32; % Number of neurns.
parameters = struct;

% Initialize weights/biases for input layer
sz = [numNeurons 1]; % here 1 represent number of input variables
parameters.fc1_Weights = initializeHe(sz,2,'double');
parameters.fc1_Bias = initializeZeros([numNeurons 1],'double');

% Initialize weights/biases for hidden layers
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name + "_Weights") = initializeHe(sz,numIn,'double');
    parameters.(name + "_Bias") = initializeZeros([numNeurons 1],'double');
end

% Initialize weights/biases for output layers
sz = [1 numNeurons]; % here 1 represent number of output variables
numIn = numNeurons;
parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,'double');
parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');
disp('NN parameters are initiated.')

%% NEURAL NETWORK TRAINING
No_Iterations = 20000;
dlT = dlarray(dataT','CB');
dlT0 = dlarray(T0,'CB');
dlX0 = dlarray(X0,'CB');
disp('Deep learning arrays created.')
    
if strcmp(method,'LBFGS')
    % Options: https://au.mathworks.com/help/optim/ug/fmincon.html#input_argument_options
    options = optimoptions('fmincon', ...
        'HessianApproximation','lbfgs', ...
        'MaxIterations',No_Iterations, ...
        'PlotFcn','optimplotfirstorderopt',...
        'Display','iter',...
        'MaxFunctionEvaluations',1.2*No_Iterations, ...
        'OptimalityTolerance',1e-6, ...
        'SpecifyObjectiveGradient',true);

    [parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
    parametersV = extractdata(parametersV);
    disp('Under optimization...')
    if strcmp(NN_choice,'NN')
        objFun = @(parameters) objectiveFunctionNN(parameters,dlT0,dlX0,parameterNames,parameterSizes);
        [parametersV,fval] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
    else
        objFun = @(parameters) objectiveFunctionPINN(parameters,dlT,dlT0,dlX0,parameterNames,parameterSizes);
        [parametersV,fval] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
    end
    % Note: This is fmincon based and cannot be run on GPU at present. 
    disp('Optimization completed.')

elseif strcmp(method,'ADAM')
    numEpochs = 5000;
    miniBatchSize = numEpochs*length(dataT)/No_Iterations;
    executionEnvironment = "cpu"; % cpu, gpu
    initialLearnRate = 0.01;
    decayRate = 0.005;
    mbq = minibatchqueue(ds, ...
        'MiniBatchSize',miniBatchSize, ...
        'MiniBatchFormat','BC', ...
        'OutputEnvironment',executionEnvironment);
    if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
        dlT0 = gpuArray(dlT0);
        dlX0 = gpuArray(dlX0);
    end

    averageGrad = [];
    averageSqGrad = [];
    
    if strcmp(NN_choice,'PINN')
        accfun = dlaccelerate(@modelGradientsPINN);
    else
        accfun = dlaccelerate(@modelGradientsNN);
    end

    figure();C = colororder;grid on; plottools on; box on;
    set(gcf, 'color', 'white');set(gca,'fontsize',16);
    set(gca, 'FontName', 'Times New Roman');
    lineLoss = animatedline('Color',C(2,:));
    xlabel("Iteration");ylabel("Loss")
    ylim([0 inf]);

    start = tic;
    iteration = 0;

    for epoch = 1:numEpochs
        reset(mbq);

        while hasdata(mbq)
            iteration = iteration + 1;

            dlT = next(mbq);

            % Evaluate the model gradients and loss using dlfeval and the
            % modelGradients function.
            
            if strcmp(NN_choice,'PINN')
                [gradients,loss] = dlfeval(accfun,parameters,dlT,dlT0,dlX0);
            else
                [gradients,loss] = dlfeval(accfun,parameters,dlT0,dlX0);
            end
    
                

            % Update learning rate.
            learningRate = initialLearnRate / (1+decayRate*iteration);

            % Update the network parameters using the adamupdate function.
            [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                averageSqGrad,iteration,learningRate);
        end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);

    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
    end
  
else
    error('Please define the method name correctly.')
end

% save('ADAM_PINN_20k','parameters')

%% TEST THE SOLUTION

tTest = linspace(0,T,500);

% predict with analytical solution
x_Exact = oscillator(tTest);

% Predict with trained ML model
dltTest = dlarray(tTest,'CB');
x_NN = extractdata(model(parameters,dltTest));

% Difference between exact and approximated
xDiff = x_Exact - x_NN;

figure(); hold on; box on; plottools on;LW = 'LineWidth';
set(gcf, 'color', 'white');set(gca,'fontsize',16);
set(gca, 'FontName', 'Times New Roman');
xlabel('$t$','Interpreter','Latex');ylabel('$x(t)$','Interpreter','Latex')
plot(t,x,'k-',LW,2,'Display','Exact solution')
plot(T0,X0,'o','MarkerSize',10,'Display','Training data',LW,2)
plot(tTest,x_NN,'b-.',LW,2,'Display',[NN_choice '-' method])
plot(tTest,xDiff,'r.','Markersize',10,LW,2,'Display','Difference')
legend('Interpreter','latex')



%% Functions
   
function y = oscillator(x)
global d w0
% Source: https://beltoforion.de/en/harmonic_oscillator/
w = sqrt(w0^2 - d^2);
phi = atan(-d/w);
A = 1/(2*cos(phi));
envelop = exp(-d*x); % Damping Envelop
yn  = 2*A.*cos(phi+w*x); % Undamped Signal
y = yn.*envelop; % Damped Signal
end

function parameter = initializeHe(sz,numIn,className)

arguments
    sz
    numIn
    className = 'single'
end
% rng(123); % to fix the seed
parameter = sqrt(1/numIn) * randn(sz,className);
parameter = dlarray(parameter);

end

function parameter = initializeZeros(sz,className)

arguments
    sz
    className = 'single'
end

parameter = zeros(sz,className);
parameter = dlarray(parameter);

end

function [parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters)
% parameterStructToVector converts a struct of learnable parameters to a
% vector and also returns the parameter names and sizes.

% Parameter names.
parameterNames = fieldnames(parameters);

% Determine parameter sizes.
numFields = numel(parameterNames);
parameterSizes = cell(1,numFields);
for i = 1:numFields
    parameter = parameters.(parameterNames{i});
    parameterSizes{i} = size(parameter);
end

% Calculate number of elements per parameter.
numParameterElements = cellfun(@prod,parameterSizes);
numParamsTotal = sum(numParameterElements);

% Construct vector
parametersV = zeros(numParamsTotal,1,'like',parameters.(parameterNames{1}));
count = 0;

for i = 1:numFields
    parameter = parameters.(parameterNames{i});
    numElements = numParameterElements(i);
    parametersV(count+1:count+numElements) = parameter(:);
    count = count + numElements;
end

end

function [loss,gradientsV] = objectiveFunctionNN(parametersV,dlT0,dlX0,parameterNames,parameterSizes)

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

% Evaluate model gradients and loss.
[gradients,loss] = dlfeval(@modelGradientsNN,parameters,dlT0,dlX0);

    
% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);

end

function [loss,gradientsV] = objectiveFunctionPINN(parametersV,dlT,dlT0,dlX0,parameterNames,parameterSizes)

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

% Evaluate model gradients and loss.
[gradients,loss] = dlfeval(@modelGradientsPINN,parameters,dlT,dlT0,dlX0);

    
% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);

end

function [gradients,loss] = modelGradientsPINN(parameters,dlT,dlT0,dlX0)
global mu k 

% Make predictions with the initial conditions.
X = model(parameters,dlT);

% Calculate derivatives with respect to X and T.
gradientsX = dlgradient(sum(X,'all'),dlT,'EnableHigherDerivatives',true);
Xt = gradientsX;

% Calculate second-order derivatives with respect to X.
Xtt = dlgradient(sum(Xt,'all'),dlT,'EnableHigherDerivatives',true);

% Calculate physical loss
f =  Xtt + mu*Xt + k*X;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);
lossF = lossF*(1E-4); % the factor is provided to regularize the lossF 

% Calculate lossU. Enforce initial and boundary conditions.
dlX0Pred = model(parameters,dlT0);
lossU = mse(dlX0Pred, dlX0);


% physics and traning data loss
loss = lossF + lossU;


% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end

function [gradients,loss] = modelGradientsNN(parameters,dlT0,dlX0)

% Calculate lossU. Enforce initial and boundary conditions.
dlX0Pred = model(parameters,dlT0);
lossU = mse(dlX0Pred, dlX0);

% traning data loss
loss = lossU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end

function dlU = model(parameters,dlT)

numLayers = numel(fieldnames(parameters))/2;

% First fully connect operation.
weights = parameters.fc1_Weights;
bias = parameters.fc1_Bias;
dlU = fullyconnect(dlT,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    dlU = tanh(dlU);

    weights = parameters.(name + "_Weights");
    bias = parameters.(name + "_Bias");
    dlU = fullyconnect(dlU, weights, bias);
end

end

function parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes)
% parameterVectorToStruct converts a vector of parameters with specified
% names and sizes to a struct.

parameters = struct;
numFields = numel(parameterNames);
count = 0;

for i = 1:numFields
    numElements = prod(parameterSizes{i});
    parameter = parametersV(count+1:count+numElements);
    parameter = reshape(parameter,parameterSizes{i});
    parameters.(parameterNames{i}) = parameter;
    count = count + numElements;
end

end



    