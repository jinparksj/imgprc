%% VIDEO
% Create Video File Reader and the Video Player
videoFileLeft = 'left 2.mp4';
videoFileRight = 'right 2.mp4';

readerLeft = vision.VideoFileReader(videoFileLeft, 'VideoOutputDataType', 'uint8');
readerRight = vision.VideoFileReader(videoFileRight, 'VideoOutputDataType', 'uint8');
player = vision.DeployableVideoPlayer();

while ~isDone(readerLeft) && ~isDone(readerRight)
    frameLeft  =  readerLeft.step();
    frameRight = readerRight.step();
    [frameLeftRect, frameRightRect] = rectifyStereoImages(frameLeft, frameRight, stereoParams);
    frameLeftGray  = rgb2gray(frameLeftRect);
    frameRightGray = rgb2gray(frameRightRect);

    disparityMap = disparity(frameLeftGray, frameRightGray);
    depthMap = 120*15./disparityMap;
    step(player, depthMap);
end
%% JUST ONE IMAGE 
% Read and Rectify Video Frames

[frameLeftRect, frameRightRect] = rectifyStereoImages(frameLeft, frameRight, stereoParams);

% Compute Disparity
frameLeftGray  = rgb2gray(frameLeftRect);
frameRightGray = rgb2gray(frameRightRect);

disparityMap = disparity(frameLeftGray, frameRightGray);
figure(1);
subplot(1,2,1);
imshow(frameLeftRect);
subplot(1,2,2);
imshow(frameRightRect);
figure(2)
imshow(disparityMap, [0, 64]);
title('Disparity Map');
colormap jet
colorbar

