%%   Saliency Driven Region-Edge-based Top Down Level Set Evolution Function
%   _______________________________________________________________________
%   Le Duc Khai
%   Bachelor in Biomedical Engineering
%   FH Aachen - University of Applied Sciences, Germany.
%
%   Released on 08.03.2019.
%
%   The proposed algorithm creates active contour based on Level Set Evolution and
%   saliency principles without generating detectors. It works well with multi-color
%   images as long as objects detected possess same colors and background is homogeneous.
%
%   Implementation is based on this scientific paper:
%       Xu-Hao Zhi, Hong-Bin Shen
%       "Saliency driven region-edge-based top down level set evolution reveals the asynchronous focus in image segmentation"
%       Pattern Recognition, Volume 80, August 2018, Elsevier 
%       DOI: 10.1016/j.patcog.2018.03.010
%
%   The following codes are implemented only for PERSONAL USE, e.g improving
%   programming skills in the domain of Image Processing and Computer Vision.
%   If you use this algorithm, please cite the paper mentioned above to support
%   the authors.
%
%   Parameters:
%       image: the input image
%       timestep: the time step
%       mu: coefficient of the distance regularization term R(phi)
%       max_iter: number of maximum iterations
%       lambda: coefficient of the weighted length term L(phi)
%       alpha: coefficient of the weighted area term A(phi)
%       beta: coefficient of penalization of the length of contour L(phi)
%       epsilon: width of Dirac Delta function      
%       sigma: standard deviation of Gaussian distribution
%
%   Examples:
%       image = imread('Input/blood cells.jpg');
%       timestep = 1;  % time step
%       mu = 0.2;  
%       max_iter = 30;
%       lambda = 0.01; 
%       alpha = 0.06;  
%       beta = 5;
%       epsilon = 1.5; 
%       sigma = 0.8;
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function SDREL_segment(image, timestep, mu, lambda, alpha, beta, epsilon, max_iter, sigma)
%% Read the input image
% If color space is hsv, remember to convert the image into rgb
if ischar(image) == 1
    image = imread(image);
    original_image = image;
else 
    original_image = image;
end
if ndims(image) == 3
    image = rgb2gray(original_image);
else 
    warning('Only RGB-image is accepted');
end
img = double(original_image); % "image" is for gray-scale, "img" is for rgb
[rows cols] = size(image);
figure(1); 
subplot(3,2,1); imshow(original_image); title('Original image');
subplot(3,2,2); imshow(image); title('Grayscale image');

%% Set input parameters
lambda1 = lambda; 
lambda2 = lambda; % lambda2 can be varied
alpha1 = alpha;
alpha2 = alpha; % alpha2 can be varied
T1 = rows*cols*0.005; % Threshold is given by authors based on experiments
T2 = rows*cols*0.001; % Threshold is given by authors based on experiments

%% Edge indicator with Gaussian filter (Equation 9)
image_smooth = double(imgaussfilt(double(image), sigma, 'FilterSize', 3));
subplot(3,2,3); imshow(image_smooth, []); title('Gaussian-smoothed image');
[dx dy] = gradient(image_smooth);
g = 1./(1 + (dx.^2 + dy.^2));  
subplot(3,2,4); imshow(g); title('Edge indicator g');

%% Initialize Level Set Function (Equation 30)
c0=1;
phi_0 = c0*ones(rows, cols);
for i=1:rows
    for j=1:cols
        phi_0(i,j) = c0*sin(pi*rows*(i-1)/1500)*sin(pi*cols*(j-1)/1500);
    end
end
subplot(3,2,5); imshow(phi_0); title('Initial phi matrix');
phi = phi_0*-1;
n = 0;
figure(2);
subplot(1,2,1);
imagesc(original_image,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r','LineWidth',2);
title(['Zero level contour at frame: ' num2str(n)]);
subplot(1,2,2);
mesh(phi);
view([-55+180 55]); colormap(winter); hold on; contour(phi, [0,0], 'r','LineWidth',2);
title(['Phi result at frame: ' num2str(n)]);

%% Saliency map (Equation 15)
saliency = Saliency(img); % Equation 15, 16
saliency = saliency*255;
figure(1);
subplot(3,2,6); imshow(saliency); title('Saliency map');

%% Main loop
flag = 0; % If flag = 1, color difference will be 0 and jump to stage-two level set evolution
for n = 1:max_iter
    %% Calculate mean values of saliency (Equation 17, 18)
    s1 = 0;
    s2 = 0;
    count_in = 0;
    count_out = 0;
    for i = 1:rows
        for j = 1:cols
            if phi(i,j) >= 0
                count_in = count_in + 1;
                s1 = s1 + double(saliency(i,j));
            else
                count_out = count_out + 1;
                s2 = s2 + double(saliency(i,j));           
            end
        end
    end
    s1 = s1/count_in;
    s2 = s2/count_out;
    
    %% Calculate mean values of image intensity (Equation 12, 13)
    m1 = [0 0 0];
    m2 = [0 0 0];
    count_in = 0;
    count_out = 0;
    for i = 1:rows
        for j = 1:cols
            if phi(i,j) >= 0
                count_in = count_in + 1;
                m1(1) = m1(1) + img(i,j,1);  
                m1(2) = m1(2) + img(i,j,2);
                m1(3) = m1(3) + img(i,j,3);
            else
                count_out = count_out + 1;
                m2(1) = m2(1) + img(i,j,1);
                m2(2) = m2(2) + img(i,j,2);
                m2(3) = m2(3) + img(i,j,3);
            end
        end
    end
    m1 = m1/count_in;
    m2 = m2/count_out;
    
    %% Stage-one level set evolution (the first 4 terms of equation 25)
    alpha1 = alpha1*0.1; % Change parameters a little bit according to paper experiments
    alpha2 = alpha2*0.1;
    I_m1_sqdif = zeros(rows, cols);
    I_m2_sqdif = zeros(rows, cols);
    color_dif = ones(rows, cols);
    for i = 1:rows
        for j = 1:cols
            for k = 1:3
                I_m1_sqdif(i,j) = I_m1_sqdif(i,j) + (img(i,j,k) - m1(k)).^2;
                I_m2_sqdif(i,j) = I_m2_sqdif(i,j) + (img(i,j,k) - m2(k)).^2;
            end
            color_dif(i,j) = -lambda1*I_m1_sqdif(i,j) + lambda2*I_m2_sqdif(i,j) - alpha1*(saliency(i,j)-s1).^2 + alpha2*(saliency(i,j)-s2).^2;            
        end
    end
    if flag == 1 || n>10
        color_dif = 0;
    end
    
    %% Stage-two level set evolution (the last 2 terms of equation 25)
    [vx vy] = gradient(g);
    diracPhi = Dirac(phi,epsilon); % Equation 29
    phi_prev = im2bw(phi,0); 
    % Check boundary conditions
    phi = NeumannBoundCond(phi);    
    % Calculate differential of length term 
    [phi_x phi_y] = gradient(phi);
    s = sqrt(phi_x.^2 + phi_y.^2);
    Nx = phi_x./(s + 1e-10); % add a small positive number to avoid division by zero
    Ny = phi_y./(s + 1e-10);
    edgeTerm = diracPhi.*(vx.*Nx + vy.*Ny) + diracPhi.*g.*div(Nx,Ny); % Product rule of divergence operator    
    % Calculate differential of area term 
    areaTerm = diracPhi.*g;    
    % Calculate differential of regularized term 
    distRegTerm = 4*del2(phi);    
    % Calculate evolution equation
    phi = phi + timestep*(mu*distRegTerm + beta*edgeTerm + areaTerm.*color_dif);
    
    %% Thresholds for evolution
    phi_after = im2bw(phi,0);
    value_dif = sum(abs(phi_prev - phi_after), 'all');
    % Threshold T1
    if value_dif < T1
        flag = 1;
    end
    % Continue drawing contour
    pause(0.1)
    figure(2);
    subplot(1,2,1);
    imagesc(original_image,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r','LineWidth',2);
    title(['Zero level contour at frame: ' num2str(n)]);
    subplot(1,2,2);
    mesh(phi);
    view([-55+180 55]); colormap(winter); hold on; contour(phi, [0,0], 'r','LineWidth',2);
    title(['Phi result at frame: ' num2str(n)]);
    % Threshold T2
    if value_dif < T2 && flag == 1
        break;
    end
      
end

%% Show segmented image result
figure(3);
imshow(im2bw(phi,0));
title('Segmented image result');

%% Level set evolution component functions
function g = NeumannBoundCond(f)
% Make a function satisfy Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]); 
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);    
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  
end

function f = Dirac(x, sigma)
% Generate Dirac function
f = (1/2/sigma)*(1+cos(pi*x/sigma));
b = (x<=sigma) & (x>=-sigma);
f = f.*b;
end

function f = div(nx,ny)
[nxx,junk]=gradient(nx);  
[junk,nyy]=gradient(ny);
f=nxx+nyy;
end

end

