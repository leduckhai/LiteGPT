%% Demo 1: Blood Cells
image = imread('Input/blood cells.jpg');
timestep = 1;  % time step
mu = 0.2;  % coefficient of the distance regularization term P(phi)
max_iter = 30;% the maximum iteration time
lambda = 0.01; % weight of color intensity energy inside zero level set
alpha = 0.06;  % weight of saliency energy inside zero level set
beta = 5;% coefficient of penalization of the length of contour L(phi)
epsilon = 1.5; % parameter that specifies the width of the DiracDelta function
sigma = 0.8;
SDREL_segment(image, timestep, mu, lambda, alpha, beta, epsilon, max_iter, sigma);

%% Demo 2: X-ray Hand
image = imread('Input/Xray hand.jpg');
timestep = 1;  % time step
mu = 0.2;  % coefficient of the distance regularization term P(phi)
max_iter = 7;% the maximum iteration time
lambda = 0.01; % weight of color intensity energy inside zero level set
alpha = 0.06;  % weight of saliency energy inside zero level set
beta = 5;% coefficient of penalization of the length of contour L(phi)
epsilon = 1.5; % parameter that specifies the width of the DiracDelta function
sigma = 1.2;
SDREL_segment(image, timestep, mu, lambda, alpha, beta, epsilon, max_iter, sigma);

%% Demo 3: Paramecium
image = imread('Input/Paramecium.jpg');
timestep = 1;  % time step
mu = 0.2;  % coefficient of the distance regularization term P(phi)
max_iter = 20;% the maximum iteration time
lambda = 0.01; % weight of color intensity energy inside zero level set
alpha = 0.06;  % weight of saliency energy inside zero level set
beta = 5;% coefficient of penalization of the length of contour L(phi)
epsilon = 1.5; % parameter that specifies the width of the DiracDelta function
sigma = 0.8;
SDREL_segment(image, timestep, mu, lambda, alpha, beta, epsilon, max_iter, sigma);