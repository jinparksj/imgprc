%matlab and DoG in SIFT

clear all;
clc;
row = 512;
column = 512;
img = imread('box.jpg');
img = imresize(img, [row, column]);
img = rgb2gray(img); %change RGB to Grayscale
img = im2double(img); %increase precision of scale
original_img = img

%Local Maxima/Minima Detection

seed_sigma = sqrt(2); %base of sigma, standard deviation
octave = 3; %three octave for different size of blurred images
level = 3; %number of different blurred image
space_octave = cell(1, octave); %space of octave saving each different blurred image

for i=1:octave %make octave sapce
	space_blurred = zeros(row * 2^(2-i) + 2, column*2^(2-i) + 2, level) %level: number of blurred image
	D(i) = mat2cell(space_blurred, row * 2^(2-i) + 2, column * 2^(2-i) + 2, level);
end

temp_img = kron(img, ones(2)); %make img's one pixel as [1 1; 1 1]
temp_img = padarray(temp_img, [1, 1], 'replicate'); %padding, each one element to column and row
figure(2)
subplot(1, 2, 1);
imshow(original_img); %show original image

%DoG Pyramid
for i = 1:octave %each octave(same size of image) has different blurred image with different sigma
	temp_D = D{i}; %octave space, D{1}, D{2},...
	for j = 1:level %different blurred image with different sigma
		sigma = seed_sigma * sqrt(2)^(1/level)^((i-1) * level + j); %by David????
		p = (level) * (i-1); %different octave indext
		
		figure(1); %show different octave and different blurred image
		
		f = fspecial('gaussian', [1, floor(5*sigma)], sigma);
		
		L1 = temp_img; %original image

		if( i == 1 && j == 1) % it is the first of blurred image in octave, so for DoG, it 
			L2 = conv2(temp_img, f, 'same'); %padded image, convolute with gaussian, same temp_img size, first gaussian
			L2 = conv2(L2, f, 'same'); %one more convolution with gaussian, second gaussian
			temp_D(:,:,j) = L2 - L1; % Difference of Gaussian 
			int_temp_D = uint8( 255 * mat2gray(temp_D(:, :, j))); %convert temp_D(double) to integer and grayscale image 
			imshow(int_temp_D);
			L1 = L2;
		else % i is not 1, j is not 1
			L2 = conv2(temp_img, f, 'same');
			L2 = conv2(L2, f, 'same');
			temp_D(:, :, j) = L2 - L1;
			if (j == level) %if it is last image
				temp_img = L1(2: end-1, 2:end-1); %padding
			end
			int_temp_D = uint8( 255 * mat2gray(temp_D(:, :, j)));
			imshow(int_temp_D);
		end
	end
	
    D{i} = temp_D;
    temp_img = temp_img(1:2:end, 1:2:end);
    temp_img = padarray(temp_img, [1, 1], 'both', 'replicate');
	
end
