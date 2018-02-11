%matlab and DoG in SIFT

clear all;
clc;
row = 512;
column = 512;
img = imread('box.jpg');
img = imresize(img, [row, column]);
img = rgb2gray(img); %change RGB to Grayscale
img = im2double(img); %increase precision of scale
original_img = img;
% Local Maxima/Minima Detection through DoG

seed_sigma = 1 / sqrt(2); %base of sigma, standard deviation
octave = 4; % octaves for different size of blurred images
level = 5; %number of different blurred image
space_octave = cell(1, octave); %space of octave saving each different blurred image

for i=1:octave %make octave sapce
	space_blurred = zeros(row * 2^(2-i) + 2, column*2^(2-i) + 2, level); %level: number of blurred image
	D(i) = mat2cell(space_blurred, row * 2^(2-i) + 2, column * 2^(2-i) + 2, level);
end

double_img = kron(img, ones(2)); %make img's one pixel as [1 1; 1 1]
double_img = padarray(double_img, [1, 1], 'replicate'); %padding, each one element to column and row
figure(2)
subplot(1, 2, 1);
imshow(original_img); %show original image

%DoG Pyramid
for i = 1:octave %each octave(same size of image) has different blurred image with different sigma
	temp_D = D{i}; %octave space, D{1}, D{2},...
	for j = 1:level %different blurred image with different sigma, 5 blurred image in each octave
        
        sigma = seed_sigma * (sqrt(2)^(j-1)) * (2 ^ (i-1))  ; %initial sigma(scale parameter) 1/sqrt(2) and keep multiplying by sqrt(2) in one octave. Next octave has 2x sigma by SIFT
        
		p = (level) * (i-1); %different octave index
		
		figure(1); %show different octave and different blurred image
		subplot(octave, level, p+j);
        
		f = fspecial('gaussian', [1, floor(5*sigma)], sigma);
		
		G1 = double_img; %double img and initial img of octave group
        
        %DoG Operation
		if( i == 1 && j == 1) % it is the first of blurred image in octave, so for DoG, double image convolution
			G2 = conv2(double_img, f, 'same'); %padded image, convolute with gaussian, same temp_img size, first gaussian for x direction
			G2 = conv2(G2, f', 'same'); %one more convolution with gaussian, second gaussian for y direction
			temp_D(:,:,j) = G2 - G1; % Difference of Gaussian 
			int_temp_D = uint8( 255 * mat2gray(temp_D(:, :, j))); %convert temp_D(double) to integer and grayscale image / normalization
			imshow(int_temp_D);
			G1 = G2;
		else % i is not 1, j is not 1
			G2 = conv2(double_img, f, 'same');
			G2 = conv2(G2, f', 'same');
			temp_D(:,:,j) = G2 - G1;
            G1 = G2;
			if (j == level) %if it is last image
				double_img = G1(2: end-1, 2:end-1); %padding
            end
			int_temp_D = uint8( 255 * mat2gray(temp_D(:,:,j)));
			imshow(int_temp_D);
        end
        
    end
    
    %an octave is done and prepare next octave
    
    D{i} = temp_D;
    double_img = double_img(1:2:end, 1:2:end); %make double image half
    double_img = padarray(double_img, [1, 1], 'both', 'replicate'); %padding
end


% Find keypoint

interval = level - 1; %interval?? 3-1 = 2
number = 0;

for i = 2:octave+1 
    number = number + (2^( i - octave ) * column) * (2 * row) * interval; %number = number of key points at each octave
end

keypoint = zeros(1, 2*number); %key point space
flag = 1; %index for keypoint


for i = 1 : octave %each octave
    [m, n, ~] = size(D{i});
    m = m - 2;
    n = n - 2;
    
%     volume = m * n / (4^(i-1)); % 1024 * 1024 or 512 * 512 / (4^1) or 256 * 256 / (4^2)
    pixel_space = m * n;
    
    for k = 2:interval %blurred images, among four images, two images are remaind with keypoints
        for j = 1:pixel_space % pixel order like cell array
            x = ceil(j / n); %ceil function: ceil(1/512) : 1 / x = 1, m / (4^(i-1))
            y = mod(j-1, m) + 1; % 1, m * n / (4^(i-1)) - 1 mod m
            
            sub = D{i} (x:x+2, y:y+2, k-1:k+1); %to find keypoint, three images are needed and 3 by 3 windowed pixels are compared
            
            maxima = max(max(max(sub)));
            minima = min(min(min(sub)));
            
            if(maxima == D{i}(x+1, y+1, k)) %if checking pixel is maxima
                temp = [i, k, j, 1]; %keypoint, 1 is maxima index [octave, DoGed image, keypointed pixel location, index of maxima/minima]
                keypoint(flag:(flag + 3)) = temp; %save found keypoint in keypoint space
                flag = flag + 4; 
            end
            
            if(minima == D{i}(x+1, y+1, k))
                temp = [i, k, j, -1]; %keypoint, 1 is minima index [octave, DoGed image, keypointed pixel location, index of maxima/minima]
                keypoint(flag:(flag +3)) = temp;
                flag = flag + 4;
            end
        end
    end
end

[m, n] = size(img);
x = floor((keypoint(3:4:end) - 1) ./ (n ./ (2 .^ (keypoint(1:4:end) - 2)))) + 1; 
y = mod((keypoint(3:4:end) - 1), m ./ (2 .^ (keypoint(1:4:end) - 2))) + 1;

ry = y ./ 2 .^(octave -2 -keypoint(1:4:end)); %need to revise
rx = x ./ 2 .^(octave -2 -keypoint(1:4:end));

figure(2);
subplot(1, 2, 2);
imshow(original_img);
hold on
plot(rx, ry, 'r+');
