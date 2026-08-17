# LiDAR-Based Vehicle Detection and Tracking for Autonomous Racing

Marcello Cellina, Matteo Corno, and Sergio Matteo Savaresi

Abstract—Autonomous racing provides a controlled environment for testing the software and hardware of autonomous vehicles operating at their performance limits. Competitive interactions between multiple autonomous racecars however introduce challenging and potentially dangerous scenarios. Accurate and consistent vehicle detection and tracking is crucial for overtaking maneuvers, and low-latency sensor processing is essential to respond quickly to hazardous situations. This paper presents the LiDAR-based perception algorithms deployed on Team PoliMOVE's autonomous racecar, which won multiple competitions in the Indy Autonomous Challenge series. Our Vehicle Detection and Tracking pipeline is composed of a novel fast Point Cloud Segmentation technique and a specific Vehicle Pose Estimation methodology, together with a variable-step Multi-Target Tracking algorithm. Experimental results demonstrate the algorithm's performance, robustness, computational efficiency, and suitability for autonomous racing applications, enabling fully autonomous overtaking maneuvers at velocities exceeding 275 km/h.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//16627775-2d6d-4e2c-9b0b-7cf6dcb9adc7/markdown_0/imgs/img_in_image_box_621_348_1126_626.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A45Z%2F-1%2F%2F5d8b73935146f5699923acd55814fbaf3376c1e9f2db4d73a43796d13b8952b4" alt="Image" width="41%" /></div>


Index Terms—Autonomous Racing, Point Cloud Segmentation, Vehicle Detection, L-Shape Fitting, Multi-Target Tracking (MTT).

## I. INTRODUCTION

<div style="text-align: center;"><div style="text-align: center;">Fig. 1. Team PoliMOVE's Dallara AV21 "MinerVa" defending from an autonomous overtaking maneuver initiated by TUM Autonomous Motorsport during the final race of the Indy Autonomous Challenge event on January 7, 2023, at the Las Vegas Motor Speedway Credits: Indy Autonomous Challenge.</div> </div>


AUTONOMOUS RACING allows for safe testing of an autonomous vehicle's full software and hardware stack at the limits of its performance in a controlled environment.

For the correct planning and execution of overtaking maneuvers, accurate and consistent tracking of the opponent vehicle state and future trajectory prediction are necessary [2]. At the same time, due to the high velocities involved, reducing the signal processing time and latency is fundamental to react promptly to changes in the environment.

The competitive interaction of multiple autonomous race-cars drastically increases the occurrence of challenging situations like high-speed obstacles and collision avoidance [1]. Providing this kind of testing environment is one of the main goals of the Indy Autonomous Challenge (IAC), the first multivehicle competition series for level 4 autonomous racecars.

A low-latency, robust, and computationally efficient target tracking algorithm with a high detection range is fundamental for safe and successful autonomous overtaking maneuvers. This paper presents the LiDAR-based vehicle detection and target tracking algorithm deployed on Team PoliMOVE's Dallara AV21 "MinerVa" which won first place in all three multi-vehicle IAC competitions it entered.

In this work, we build an online algorithm for reliable vehicle detection from Point Cloud data with a latency lower than the sensor refresh rate, together with the capability of fully observing the target's 2D pose, tracking its motion and estimating its linear and angular velocities, without the availability of labeled data.

To fulfill these requirements, we implemented a novel Point Cloud segmentation algorithm capable of processing in parallel the three LiDAR sensors mounted on the vehicle, a multi-hypothesis L-shape fitting technique for a racing vehicle moving on a racetrack and a Multi-Target Tracking (MTT) module, which estimates the target speed, heading and yaw rate from position measurements.

The output of this pipeline is then fed to the opponent trajectory prediction and Ego-vehicle trajectory planning module to initiate eventual overtaking or defensive maneuvers.

The main contributions of this paper lie in the following:

- A fast and efficient Point Cloud segmentation algorithm working on unstructured Point Clouds with non-uniform, time-varying scan pattern with no information loss.

- A 2D pose estimation technique for an irregularly shaped vehicle from a Point Cloud acquired on a racetrack.

• A variable-step target tracking algorithm.

All of the methodologies presented in this paper are capable of online operation as they have been tested experimentally online by performing fully autonomous overtakes on vehicles travelling at velocities superior to 250 km/h.

These contributions are presented in the paper according to the following structure: In Section II, we summarize the main research contributions and open challenges in the field of vehicle detection and tracking for autonomous driving. In Section IV, we describe the experimental vehicle used for this research work, while in Section III, we provide a top-down description of all the components of the algorithm we developed. Then in Section V, we provide a quantitative performance evaluation of every algorithmic step, together with the description of the dataset used for the analysis. Finally, we conclude our work in Section VI, by summarizing the approach used and the results achieved while discussing potential future works on this subject.

## II. RELATED WORK

This section provides a literature review of the state of the art methodologies for Vehicle Detection and Tracking, with a strong focus on using LiDAR sensors. We will divide the main problem into its principal sub-problems and analyze the main approaches used to solve them.

Figure 2 shows a taxonomy of the main research problems related to Vehicle Detection and Tracking [3] [4]. In this work, we will focus on LiDAR-based methods, which, despite lacking the velocity measurement and weather robustness of RADARs and the high range and color information of Cameras, provide the greatest position accuracy of the kind, which is crucial for close racing applications. Multi-modal fusion methods, which increase the detection and tracking performance by combining the benefits of the different sensor types, are beyond the scope of this work.

Concerning LiDAR-Based Vehicle Detection and Tracking, the dominant approach in literature is Tracking-by-Detection, which divides Vehicle Detection and Target Tracking as separate problems, as opposed to End-to-End tracking, which extracts track directly from LiDAR Point Clouds, usually with the use of Convolutional Neural Networks (CNNs). Although in recent years there has been an increase of research in End-to-End Tracking and Target Tracking methods, the use of Convolutional Neural Networks (CNNs) has also been proposed.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//16627775-2d6d-4e2c-9b0b-7cf6dcb9adc7/markdown_1/imgs/img_in_image_box_101_1043_587_1397.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A45Z%2F-1%2F%2Fd49ffdcc1ffde05df07e861fc92b7e90be00804ff0897ca56ee569b82da59d9c" alt="Image" width="39%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 2. Representation of the main research problems in Vehicle Detection and Tracking, the most common solutions in literature and their relationship. In bold, the research problems, in italic the approaches used in this work. Neighboring boxes should be considered as alternative methodologies.</div> </div>


to-End tracking as in [5], [6] and [7], they are not established methods and have yet to prove their robustness and reliability for online use.

A similar division between algorithmic and data-driven approaches lies in the Vehicle Detection and Pose Estimation problem: although historically treated as separate problems, there has been a growing interest in scientific literature towards CNN-based Vehicle Detection networks, like [8], [9] and other approaches mentioned in [10].

The development of these data-driven algorithm relies on public, labeled datasets such as KITTI [11], NuScenes [12], and Waymo [13]. The vehicles and environments constituting these datasets, however, have very few similarities with our application. To address this gap, the RACECAR dataset [14] has been published by leveraging the contributions of multiple IAC teams, although no labelled version is available at the moment of writing.

Furthermore, the poor generalization capabilities of most data-driven perception algorithms may lead to unsatisfactory behaviour in many edge cases often encountered in racing, like the presence of debris on the track, or a vehicle emitting smoke or spinning out of control. Sharing the considerations expressed in [15], we decided that data-driven methods were not suitable for this application, as they are difficult to transfer from synthetic data to real life.

A separate distinction in Tracking-by-Detection methods lies in the detection mechanism: In Detection-by-Segmentation approaches, vehicles are isolated from the surrounding environment by grouping points in the LiDAR point cloud based on geometric features. In contrast, Detection-by-Motion relies on detecting changes in the position of objects over time, using temporal differences between consecutive point cloud frames to identify moving objects. The latter methods usually rely on a representation of the surrounding environment in terms of occupancy, like the Virtual Scan of [16] or the Octree in [17]. Although simpler and potentially better performing in cluttered environments than Detection-by-Segmentation methods, these methods are incompatible with static obstacles, and therefore would pose a safety hazard in the presence of stopped vehicles on the racetrack.

For these reasons, in this work and in the rest of this literature review, we will focus on a Tracking-by-Detection approach, composed by three main algorithmic steps: Point Cloud Segmentation, Vehicle Pose Estimation and Target Tracking. We will delve into each of these sub-problems in the following sub-sections.

### A. Point Cloud Segmentation

Point Cloud segmentation is the process of partitioning a set of 3D points into meaningful groups or segments based on their characteristics or spatial relationships. This process usually facilitates subsequent operations such as object classification and vehicle pose estimation.

A special case of Segmentation is Ground Removal, which refers to the detection of points belonging to the ground plane. This step is often applied as a pre-processing step for the proper Segmentation algorithm.

In literature, the ground plane has been filtered out by processing the Point Cloud in polar coordinates, using the polar binning method proposed in [18] or the height difference method proposed in [16], and later expanded by [19] with the addition of a smoothing filter. However, all these methods work only for mechanical LiDAR sensors, and are not suited for modern MEMS scanning LiDARs.

Concerning autonomous racing, a 2.5D grid approach as in [20] allows for great computational efficiency, as all the conditions are applied to properties of the grid. However, this method lacks robustness to outliers. A more sophisticated approach is presented in [21], which employs a variation of the algorithm presented in [22] to label the ground points. The main limitations of grid-based approaches are the performance degradation at high range and the inability to process multiple Point Clouds in parallel.

Beyond Ground Removal, the most established Point Cloud Segmentation method is the Euclidean Clustering algorithm provided by the PCL library in [23], whose computational complexity, however, makes it unsuitable for online use with large Point Clouds without heavy input downsampling, and subsequent information loss.

A mixed approach is employed in [18], with most of the segmentation performed by searching for connected components on a 2D grid, which can be performed efficiently using image processing techniques, and then a 3D voxel grid is built only for those clusters that need additional segmentation. A similar voxel grid approach is used by [24]. However, these methods share the information loss problem of similar 2.5D approaches.

Higher-performing methods try to exploit the structure of the LiDAR scan in order to segment the Point Cloud, usually working in a mix of spherical and Cartesian reference frames. It is the case of the Scan-Line Run algorithm presented in [25], which first segments the points belonging to a single Point Cloud layer and then looks for correspondences between contiguous layers, or of [26], which employs a Range Image representation of the Point Cloud to perform the Segmentation. These methods however rely heavily on the structured Scan Pattern of rotating LiDAR Sensors.

A combination of a Range Image representation of the Point Cloud and a CNN has been presented in [27], providing superior performance compared to classical algorithmic methods. A similar approach based on a cylindrical projection, which maintains the 3D information that is lost during the range image projection, is used in [28]. These methods however share the drawbacks expressed in the previous considerations about CNN-based algorithms, making them unsuited for autonomous racing.

### B. Vehicle Pose Estimation

The goal of Vehicle Pose Estimation in autonomous driving is to infer the vehicle position and orientation, as well as its dimensions, from its corresponding LiDAR points. The two prevalent methodologies are L-shape Fitting and Box Fitting.

L-shape Fitting, originally developed for 2D LiDARs and then expanded to 3D sensors, aims to model the partial observability due to self-occlusions of rectangular shaped vehicles as passenger cars. These methods use either a variation of the Ramer-Douglas-Peucker algorithm $ ^{[29]} $ to find the cluster's primary orientation, as presented in [30] and in [31], or a RANSAC-based approach as in [32] and [33], and then proceed to fit the minimum area rectangle given its principal orientation. Although very computationally efficient, these methods have a strong reliance on the hypothesis of rectangular shape of the tracked vehicles, which is violated in autonomous racing applications, and are not generally robust to outliers.

Box Fitting techniques instead consist in estimating a bounding box around the entire vehicle by means of the minimization of a cost function, like in [34], [35] and in [36]. Although more robust to outliers than L-shape fitting techniques, as they do not rely explicitly on the rectangular shape on the tracked vehicle, they have a high variance in estimation, and they have not been tested with vehicles other than road cars or trucks.

### C. Target Tracking

Target Tracking is the process of estimating the number of objects of interest (targets) present in the tracking volume, together with their partially measurable states. When applied to Autonomous Ground Vehicles (AGV) usually involves estimating the number of moving vehicles surrounding the EGO vehicle, their 2D pose and linear and angular velocity.

Considering the taxonomy shown in Figure 2, together with the definitions expressed in [37], the first division of Tracking algorithms lies in the difference between Single Object Tracking (SOT), which is employed in single-objective control systems, and Multiple Object Tracking (MOT) techniques, which are more appropriate to autonomous driving applications, where the number of targets is unknown and can be larger than 1.

MOT problems can be further divided into Multi-Target Tracking (MTT) problems, where each target is supposed to generate at most one measurement at each iteration, [38], and Extended Target Tracking (ETT) problems, where each target can generate more than one measurement, as it is usually the case with LiDAR sensors. ETT techniques for vehicle tracking are usually based on strong assumptions about the vehicle shape and should guarantee a larger precision in tracking than MTT, although some shape-free ETT techniques exist as in [39]. Although ETT techniques have been successfully applied to autonomous driving [40] [41], their strong assumptions about the rectangular shape of the target make them unsuitable for tracking a racecar.

All Target Tracking algorithms use a dynamical filter in order to estimate the target state and approximate its uncertainty distribution. The most common filters employed are the Extended Kalman Filter (EKF), the Unscented Kalman Filter (UKF), the Particle Filter (PF) and the Interacting Multiple Models (IMM) estimators.

The classical approach to the Multi-Target Tracking problem involves the use of an Extended Kalman Filter (EKF) as in [35], which has also successfully been applied to autonomous

racing in [42]. Although simple, this filter has proven to be effective in order to plan and execute autonomous overtaking maneuvers at high speed.

A more sophisticated approach involves the use of multiple EKF estimators in the Interacting Multiple Models (IMM) framework. [20] presents an autonomous racing application of a IMM-UKF-PDAF tracker, although it does not provide a quantitative performance evaluation of the algorithm performance. In [16], the authors employ a PF to track the moving vehicles, but the higher computational load intrinsic in the PF does not scale well with the number of targets in the scene.

The last aspect of Target Tracking algorithms lies in the mechanism for Data Association, which can be a simple Global Nearest Neighbor (GNN) association, based on the minimization of a cost function like in [43] or a more computationally intensive Joint Probabilistic Data Association (JPDA) algorithm as in [31].

### D. Open Challenges

This section has provided an overview of the key subproblems involved in the LiDAR-based Vehicle Detection and Tracking for Autonomous racing. Most of the scientific literature on this topic aims to solve a subset of the problems schematized in Figure 2 under a variety of operational environments, although mostly focused on urban driving.

The most established Point Cloud Segmentation algorithms have large computational latencies that make them unsuitable for online use at high speed without significant information loss due to downsampling. The few methods designed for high computational efficiency suffer from lack of robustness and generality as they work under very restrictive assumptions. To the best of the authors' knowledge, there is no efficient, general and highly performing Point Cloud Segmentation algorithm available in scientific literature.

This work targets the research gap concerning fast and efficient Point Cloud Segmentation algorithms with broader domain assumptions than those available in literature, together with a novel solution to the Vehicle Pose Estimation problem applied to Autonomous Racing.

Concerning the Vehicle Pose Estimation problem, the scientific literature proposes solutions only under the assumption of the rectangular shape of the vehicles, which is not valid in this application, and will require an ad-hoc solution. On the other hand, Target Tracking methodologies are well-established, allowing us to use state-of-the-art algorithms.

## III. METHODOLOGIES

In this section, we present our approach to solving the problem of LiDAR-Based Vehicle Detection and Tracking for Autonomous Racing. We first describe the main assumptions and algorithmic structure, and then we delve into more detail concerning the main algorithmic steps.

We propose an algorithm that operates under the following domain assumptions:

- The LiDAR sensors provide an unstructured Point Cloud, with a potentially non-uniform and time-varying scan pattern.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//16627775-2d6d-4e2c-9b0b-7cf6dcb9adc7/markdown_3/imgs/img_in_image_box_635_108_1114_394.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A46Z%2F-1%2F%2F847a0e30e9a71cc1937c7e70150a545d3fb4ca6cf86b5451016a015caf2200e2" alt="Image" width="39%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 3. Block scheme of the LiDAR-Based Tracking algorithm</div> </div>


- The LiDAR sensors provide, other than the XYZ coordinates, Intensity and Ring information for each point.

- The platform (EGO) vehicle is equipped with a number of LiDAR sensors with adjacent or overlapping Field of View.

- The size of the vehicles to be tracked is a-priori known.

- Precise EGO vehicle absolute localization is available, together with a map of the operative environment.

Figure 3 shows the algorithmic architecture of this work. The process begins with Point Cloud Segmentation, where we apply a Range Image-based Ground Removal and Clustering algorithm. This process is performed in parallel on the three input Point Clouds. We then merge the three Segmented Point Clouds, discarding clusters with size and position incompatible with a racecar. Next, we estimate the 2D pose of the vehicles using a robust method based on the information from the track layout. Finally, the Target Tracking algorithm tracks the opponent vehicles, predicts their future states, and estimates their velocities. The Target Tracking output is fed to the Local Planning module.

The problem of motion distortion in LiDAR scans, which is of great importance for mapping applications and whose effect is particularly strong in this application due to the high EGO vehicle speed, is not relevant to the Vehicle Detection and Tracking due to the small relative speed between the vehicles, and therefore was not addressed in this work.

### A. Point Cloud Segmentation

The Point Cloud segmentation algorithm presented in this work is an expansion of the Range Image Clustering method presented in [19]. A Range Image  $ \mathbf{R} $ is a 2D dense representation of a 3D Point Cloud  $ \mathbf{P} $ obtained by transforming  $ \mathbf{P} $ into spherical coordinates and projecting it on the azimuth-elevation plane. It comes in the form of a  $ n \times m $ matrix where each row  $ i \in [1, n] $ represents a constant elevation value and each column  $ j \in [1, m] $ represents a constant azimuth value. The element  $ r_{i,j} \in \mathbf{R} $ represents the measured range for a given azimuth and elevation, such that the Range Image representation of a point is equivalent to its 3D coordinates, as  $ \forall p \in \mathbb{R}^3, p \subset \mathbf{P}, p_k \leftrightarrow r_{i,j} $.

This representation works well with a rotating LiDAR. A rotating LiDAR allows for easy construction of the Range

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_0/imgs/img_in_image_box_95_112_600_188.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A46Z%2F-1%2F%2F134961381d6f127be820c59181ad98700ee58f2bccb11a67787170cc3dd6d7f1" alt="Image" width="41%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(a) Input Point Cloud P in spherical coordinates (blue: near, red: far)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_0/imgs/img_in_image_box_112_233_602_276.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A46Z%2F-1%2F%2F0c699d0c12b3693254975c896e54f543c53df7a9aac41a7bae5d0c9d52a8ccf8" alt="Image" width="40%" /></div>


(b) Range Image R (blue: near, red: far)

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_0/imgs/img_in_image_box_116_320_600_359.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A47Z%2F-1%2F%2F67570a40dc983201589a79a44a0c7f416cabec145959c544aa6f9468d7000313" alt="Image" width="39%" /></div>


(c) Angle Image A (blue: horizontal, red: vertical)

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_0/imgs/img_in_image_box_116_406_599_445.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A47Z%2F-1%2F%2F5cc7f6836d949bb02e8d301da4d211c333e42633bf3bd2c00cc23060d174518b" alt="Image" width="39%" /></div>


(d) Smoothed Angle Image  $ \hat{A} $ (blue: horizontal, red: vertical)

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_0/imgs/img_in_image_box_115_493_597_536.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A47Z%2F-1%2F%2F0314857515a8c4a605e2471969a1fbee206c2dbf42521ed85f307e31c1889f42" alt="Image" width="39%" /></div>


(e) Repaired non-Ground Range Image  $ \mathbf{R} $ (blue: near, red: far)

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_0/imgs/img_in_image_box_117_581_599_624.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A48Z%2F-1%2F%2F39d2fcbbffb008e8646d8563ddee40f2976e50fbaacae1efab93341aaf4ea4ae" alt="Image" width="39%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(f) Label Image L (color: label)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_0/imgs/img_in_chart_box_118_701_579_943.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A48Z%2F-1%2F%2Ff8cbfd606f0249f016174b7a264804eeaf647d008e3656b0678be63cb526ef3e" alt="Image" width="37%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(g) Segmented Point Cloud (color: label)</div> </div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 4. The five steps of the Point Cloud segmentation algorithm and the resulting segmented Point Cloud. Images are scaled vertically for easier visualization.</div> </div>


Image as the distribution of the scan points on the azimuth-elevation plane is fixed, as it depends on the geometrical characteristics of the sensor. This is not the case for a scanning LiDAR: due to the simultaneous horizontal and vertical motion of the sensor head, the elevation is not constant along the scan lines, as Figure 4a shows. Moreover, some sensor models allow for online configuration of the scan pattern, making this azimuth-elevation characteristic time-varying.

If the sensor provides the information about which scan layer originated every point, however, it is possible to construct a range image  $ \mathbf{R} $ on the azimuth-layer plane, where each row  $ l \in [1, n] $ corresponds to a single scan line and every element  $ r_{l,j} \in \mathbf{R} $ represents the measured range of the scan point belonging to layer  $ l $ and azimuth corresponding to column  $ j $. An example of such construction is presented in Figure 4b.

This azimuth-layer Range Image is not sufficient however to reconstruct the original image, as it is lacking information about the elevation of the points. This information is stored into an Elevation Image  $ \mathbf{E} $, a second matrix of the same size as  $ \mathbf{R} $, where each element  $ e_{l,j} \in \mathbf{E} $ contains the elevation value of the point whose range is stored in  $ r_{l,j} $. In this way, it is possible to define the mapping  $ p_{l,j} \leftrightarrow \{r_{l,j}, e_{l,j}\} $ to reconstruct the original  $ \mathbb{R}^3 $ point without loss of information.

Many LiDAR sensors have the capability of providing multiple range readings for every scan line. However, the value of every pixel of R must be unique (or empty). Thus, a policy is needed to determine which point to retain. For this application, we chose to keep the point with the largest intensity value, as it is less likely to belong to dust or debris.

On the other hand, R may figure several empty pixels, as Figure 4b shows, caused by missed scan returns especially at large distances, or an uneven spacing across the azimuth. These points are artificially filled using a mean filter across the non-empty neighbors in the same image column.

After completing the construction of R and E, following the approach presented in [26], we build the Angle Image A. It consists of a  $ n-1 \times m $ matrix, containing the angle between the x-y plane (in the sensor reference frame) and the vector connecting two points with the same azimuth belonging to consecutive layers, as in (1).

 $$ a_{i,j}\in\mathbf{A}=\arctan\left(\frac{r_{i,j}\cos(e_{i,j})-r_{i+1,j}\cos(e_{i+1,j})}{r_{i,j}\sin(e_{i,j})-r_{i+1,j}\sin(e_{i+1,j})}\right) $$ 

Figure 4c shows the result of the computation of the Angle Image A. Then, we smooth it by applying a column-wise Savitzky-Golay kernel with window size  $ s_{SG} $ to A, obtaining the Smoothed Angle Image  $ \hat{A} $ which is shown in Figure 4d. By imposing a threshold  $ th_{gnd} $ over the maximum allowed ground slope, we can label as not ground the points where  $ \hat{a}_{i,j} > th_{gnd} $. Since A/ $ \hat{A} $ has one row less than R, the bottom row is by default labelled as ground.

The resulting non-ground Range Image is then fed to the angle-based segmentation step introduced in [19], where every connected component of  $ \mathbf{R} $ is further segmented via the Breadth-First Search algorithm that compares every element of with its 4-neighbors. In particular, two points  $ p_1, p_2 \in P $ being neighbors in  $ \mathbf{R} $ are allowed in the same cluster  $ C $ if the angle  $ \beta_{1,2} $, representing the incidence of the segment connecting the two points with the segment connecting the first and the origin, is lower than a threshold  $ th_{seg} $. The angle  $ \beta_{1,2} $ is defined in (2)

 $$ \beta_{1,2}=\arctan\left(\frac{r_{1}\sin\alpha_{1,2}}{\left(r_{1}-r_{2}\right)\cos\alpha_{1,2}}\right) $$ 

where  $ \alpha_{1,2} $ represents either the azimuth difference of a pair row neighbors or the elevation difference of a pair of column neighbors. The output of this algorithm is a Label Image L whose element  $ li, j \in \mathbf{L} $ represents the univocal index of the cluster associated to the point  $ p_k \leftrightarrow r_{i,j} $

A strong limitation of the approach described in [19] lies in its inability to effectively manage a cluster that extends across multiple connected components within the range image, as the Breadth-First-Search algorithm is applied only to elements

belonging to the same connected component of the non-ground Range Image.

Usually, the front and rear suspension brackets of the opponent vehicle, composed of small and opaque carbon fiber pipes, are either not directly hit by a LiDAR ray or they provide no reflection, being black. At the same time, the cars present multiple horizontal surfaces that can be labeled as ground, like the front and rear wings and the under-body. This usually results in having multiple separated non-ground components, as shown in Figures 4c and 4d.

Therefore, we performed the Non Ground Range Image Reparation (NGRIR) after the Ground Removal step, by applying a horizontal mean filter kernel of size  $ w_{s} $ to the ground elements of  $ \mathbf{R} $, iterating only over the non-ground elements in the same row of the range image. An additional check over the maximum depth difference  $ th_{r} $ of the symmetric element pairs avoids the merging of connected components belonging to different elements, as expressed in (3).

 $$ \begin{aligned}N(r_{i,j})&=\{r_{i,j+k},k\in[-w_{s},+w_{s}]\\&\quad\|\exists r_{i,k}\wedge|r_{j-k}-r_{j+k}|<t h_{r}\}\\r_{i,j}^{*}&=\sum N(r_{i,j})/|N(r_{i,j})|\end{aligned} $$ 

The result of the NGKIR algorithm is the Reparied Non-Ground Range Image  $ \hat{R} $ shown in Figure 4f, while Figure 4g shows how this technique allows the Segmentation Algorithm to assign the same label to a cluster spanning multiple connected components in the non-ground range image.

So far, the input Point Clouds coming from the multiple LiDAR sensors have been processed in parallel on different CPU cores. When the Segmentation step is terminated, the algorithm collects the segmented Point Clouds and applies a 3D rigid transformation to bring them to a common, vehicle-fixed reference frame. Then, we perform Cluster Merging in order to obtain a single, segmented Point Cloud.

To determine which pairs of clusters  $ C_1 $ and  $ C_2 $ from two neighboring Point Clouds  $ \mathbf{P}_1 $ and  $ \mathbf{P}_2 $ belong to the same object and therefore should be merged, we apply the following equation

 $$ \begin{array}{r l}{C_{1}\equiv C_{2}}&{\mathrm{i f}\quad\exists i,j\in[1,n]\quad\parallel\quad|\mathbf{p}_{\mathbf{i,m}}^{1}-\mathbf{p}_{\mathbf{j,1}}^{2}|<t h_{m r g}}\end{array} $$ 

to the clusters belonging to the last column of $\mathbf{R}_1$ and the first column of $\mathbf{R}_2$ (for example, the last/right column of the Range Image corresponding to the Front LiDAR scan and the first/left column of the Range Image corresponding to the Right LiDAR scan), where $th_{mrg}$ is the euclidean distance threshold to consider the two points part of the same cluster. Although simple, this brute force approach works well in practice, and it proved to be robust to small errors in extrinsic sensor pose calibration.

Figure 5 shows the qualitative results of the cluster merging process. The front wheel captured by the frontal LiDAR is correctly merged with the rest of the opponent vehicle captured by the right sensor. At the same time, the segments of the wall captured in the front scan are joined with their corresponding ones from the right LiDAR, but the cluster is immediately broken as the presence of the opponent car does not allow

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_1/imgs/img_in_image_box_620_112_1126_333.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A50Z%2F-1%2F%2F379f7d75a71fe7a242558dabf9bfb6bc2de2b1b9d35113fe0c7106530b9346da" alt="Image" width="41%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(a) Input Point Clouds with intensity measurements (red: lowest, blue: highest)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_1/imgs/img_in_image_box_622_381_1126_601.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A50Z%2F-1%2F%2Fa734e1b651d8bd258f7f8d3a47d7d87701ff76e04b614f17fc98b2b6de8cca0b" alt="Image" width="41%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(b) Segmented merged Point Cloud (color: label)</div> </div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 5. Input and Segmented Point Clouds from the three sensors capturing an opponent performing an overtaking maneuver at 250 km/h. The EGO vehicle is represented as a semi-transparent 3D graphical model. Grid size: 1m.</div> </div>


the wall to form a single connected component in the Range Image of the right Cloud.

### B. Vehicle Detection and Pose Estimation

After merging the clusters across different sensors, we discard all the clusters with the longest dimension larger than the a-priori known vehicle size. Furthermore, we use the EGO vehicle pose estimation to project every cluster centroid on a global map of the track surface in order to discard all the objects lying outside the map.

Although the use of the map to classify clusters adds a risk of false negative detections in case of an imprecise EGO pose estimation, this step is fundamental to filter out false detections originating from large overhanging signs or vehicles parked in the pit lane, which do not participate in the on-track action.

For every cluster labelled as a vehicle, we want to estimate its 2D pose  $ \{x, y, \psi\}_{opp} $ from the 3D scan points. We will follow the most common method to solve this problem by fitting a rectangular model to the points.

A rectangle is defined by 5 variables:  $ \{x, y, \psi, W, L\} $, however the most important is the principal heading  $ \psi $ as, given a certain heading, finding the minimum area rectangle enclosing all the cluster points is trivial. Due to occlusions in the LiDAR scan, however, the width and length of the minimum area rectangle are smaller than the actual vehicle dimensions. For this application, the width and length of the vehicle are a-priori known, and can be imposed as constraints to the rectangle fitting problem.

Therefore, given a set of points belonging to a cluster  $ \{p_i\} \subseteq \mathbf{C} $, and a estimated heading  $ \psi_{opp} $, it is immediate to find the coordinates of the corner point  $ p_c = \{x, y\}_{c} $ by fitting

the minimum area rectangle with principal heading  $ \psi $. Once the corner point has been found, it is possible to impose the known vehicle width  $ W = W_{EGO} $ and length  $ L = L_{EGO} $, and to compute the coordinates of the vehicle geometrical center  $ \{x, y\}_{opp} $ in the EGO-centered reference frame.

We propose an innovative method for estimating  $ \psi_{opp} $ which is based on the fusion of two independent estimators: a trajectory-based heading estimation,  $ \hat{\psi}_{REF} $ and a classical L-shape fitting technique based on the bidimensional rectangle fitting error variance minimization  $ \hat{\psi}_{VAR} $.

The first method computes  $ \hat{\psi}_{REF} $ using the prior knowledge of the track map by assuming the opponent to be traveling parallel to the center line. Given the coordinates of the cluster centroids transformed in the map-fixed reference frame, it is trivial to project it over the centerline and compute its heading, which once transformed back in the vehicle-fixed reference frame becomes  $ \hat{\psi}_{REF} $. The rationale behind this estimator lies in the fact that in high speed oval racing the vehicles tend to have negligible incidence angles with respect to the track centerline, as the sideslip angles are reduced and overtake or avoidance maneuvers require little heading changes.

This method represents an open-loop estimation which, is not dependent on the real opponent vehicle heading. While the approximation  $ \psi_{opp} \approx \hat{\psi}_{REF} $ holds well under nominal racing conditions, where it can be more robust than actual L-shape estimators, it clearly fails under non-nominal conditions like a vehicle spinning without control or stopped sideways on the track surface.

The second estimator  $ \hat{\psi}_{VAR} $ aims to estimate directly the minimum area rectangle from the 3D Point Cloud following the Variance Minimization method described in [34].

The fusion of the two methods happens according to (5)

 $$ \hat{\psi}_{opp}=\begin{cases}\hat{\psi}_{REF}&if\quad|P_{REF}|\geq|P_{VAR}|\\\hat{\psi}_{VAR}&otherwise\end{cases} $$ 

where  $ P_{REF} $ and  $ P_{VAR} $ indicate respectively the cardinality of the set of points inside the estimated rectangle using  $ \hat{\psi}_{REF} $ and  $ \hat{\psi}_{VAR} $, respectively.

In this way, we are certain to always choose the most representative estimation of the opponent heading, with a preference towards the open-loop estimator when no substantial difference is present.

Figure 6 shows qualitatively the results of the double estimator: while in the first and third Cloud the difference between  $ \hat{\psi}_{REF} $ and  $ \hat{\psi}_{HAT} $ is low, and the two estimators are equivalent, in the middle Point Cloud the most correct estimation comes from  $ \hat{\psi}_{HAT} $, and it is chosen as  $ |P_{REF}| \geq |P_{VAR}| $.

### C. Target Tracking

To track the opponent vehicles, estimate their velocity, and predict their future state, we employ a Multi-Target Tracking algorithm that uses a variable-rate Extended Kalman Filter to estimate the target states with Global Nearest Neighbor data association and a three-state M/N track management logic. For a more comprehensive dissertation on Target Tracking please refer to [37].

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_2/imgs/img_in_chart_box_643_135_1066_426.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A52Z%2F-1%2F%2F2847f8d7ba488f34ec2f65284872f47513c8be5b7bf8f8284dc857bf61e59e20" alt="Image" width="34%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 6. Graphical comparison of the results of the two rectangle fitting algorithms  $ \psi_{REF} $ and  $ \psi_{VAR} $ over the segmented Point Cloud of a vehicle entering a spin.</div> </div>


The variable-step EKF update happens either at the end of the pose estimation algorithm or after a certain time since the last received measure. In case of new measurements, the time step is computed by sensor timestamp subtraction, to be robust to differences in processing latency between consequent scans. Before publishing the output, the filter performs one last prediction step to provide the most up-to-date estimation of the target state.

The model used for the tracking EKF is a Constant Velocity and Turn Rate (CVTR), also known as the Coordinated Turn (CT) model. This model is described the following equation:

 $$ \dot{X}(t)=\left\{\begin{aligned}\dot{x}(t)&=v(t)\cos(\phi(t))\\\dot{y}(t)&=v(t)\sin(\phi(t))\\\dot{v}(t)&=0\\\dot{\phi}(t)&=\omega(t)\\\dot{\omega}(t)&=0\end{aligned}\right. $$ 

and it represents the motion of a rigid body on a plane, defined by its 2D pose  $ \{x, y, \phi\} $ in the map-fixed Cartesian reference frame with constant linear and angular velocity v and  $ \omega $.

The measurements used for the filter update are the x and y centroid position transformed in the map-fixed reference frame using the EGO vehicle pose at the instant of the LiDAR frame. For this work, we decided not to include the measured heading  $ \psi $ as a filter input. The rationale behind this choice is that in critical conditions, like a vehicle spinning out of control, the course angle  $ \phi $ and the vehicle heading  $ \psi $ diverge due to the high sideslip angle. To properly handle an unreliable measure in an EKF is beyond the scope of this work.

The measurement model is linear in the state space and it is described by

 $$ \begin{aligned}Y(t)&=HX(t)\\H&=\begin{bmatrix}{{{1}}}&{{{0}}}&{{{0}}}&{{{0}}}&{{{0}}} \\{{{0}}}&{{{1}}}&{{{0}}}&{{{0}}}&{{{0}}}\end{bmatrix}\end{aligned} $$ 

The Data Association step uses the Mahalanobis Distance between the measured and tracked position weighted by the EKF output error covariance matrix S. A gating threshold is imposed over the distance to avoid wrong associations.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e4468fb4-d78e-4a71-8e2b-ea7e7245fffc/markdown_3/imgs/img_in_image_box_97_110_600_313.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A54Z%2F-1%2F%2Ff5b4d18153b4323db53be9d158af6c3fe36c0606ec9df2c53ec4bb55660c1d07" alt="Image" width="41%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 7. LiDAR sensors mounting positions and horizontal FOV coverage of the three sensors</div> </div>


defining an association region for every track. Once the distances have been computed for all the feasible association pairs, the Munkres algorithm is used in order to determine the best assignment.

The track life cycle is determined by a two threshold  $ M/N $ logic: every measure that is not associated with a track will create a new tentative track. The track state will transition to confirmed once a minimum number  $ M_{c} $ of associations have taken place in the last N iterations. In contrast, the track will become dead if less than  $ M_{d} $ iterations have taken place in the last N iterations.

## IV. EXPERIMENTAL SETUP

The platform used for this research is the Dallara AV21 shown in Figure 1, which is a level 4 autonomous vehicle based on the single-seat Dallara IL-15 chassis, with additional sensors and actuators.

This vehicle is capable of speeds up to 310 km/h in an oval circuit configuration, with 18.5 and 26.5 m/s $ ^{2} $ (2 and 2.7 g) maximum longitudinal and lateral acceleration, respectively. As the Dallara AV21 is the only vehicle admitted to the IAC competitions, these performance bounds are shared by both the ego and tracked vehicles.

The vehicles are provided with a redundant dual-antenna RTK GNSS + IMU sensor, which allows for accurate online positioning and state estimation of the ego-vehicle, while the opponent vehicle data provide precise ground truth for offline performance analysis. The onboard computer is a dSPACE Autera [44] with a 12-core CPU working at a frequency of 2.0 GHz and 128 GB of 2133 MHz RAM. It runs the Ubuntu 20.04 operating system, as our software is written in C++ using the ROS Galactic [45] middleware.

The car is equipped with three Luminar Hydra LiDAR sensors, which are the main subject of this work. The Luminar Hydra sensor [46] is a vertically scanning LiDAR with a fixed  $ 120^{\circ} $ horizontal Field-Of-View (FOV) and a software configurable vertical FOV in a subset of the  $ [-15^{\circ},+15^{\circ}] $ range.

Figure 7 shows the mounting positions of the three sensors around the cockpit of the vehicle. This configuration results in a 40 cm wide blind spot between the front and lateral sensors. The presence of the rear wing and its end plates creates a noticeable blind spot in the rearward direction, observable in Figure 5, which severely impacts the opponent detection performances in that area.

Each sensor outputs 640 scan lines per second, each one composed of around 850 points, for a total of more than 500,000 points per sensor per second. The sensor scan rate can also be configured online in the [1,30] Hz range. We found a good compromise between latency and scan density by configuring our sensor to operate at a fixed 20 Hz frequency, resulting in 32 scan lines per Point Cloud.

Every scan point is characterized, other than its cartesian coordinates, by its scan layer index, an intensity measurement, and its individual timestamp. The vertical FOV of the sensor, together with the distribution of the scan lines, can be configured online and therefore time-varying.

## V. RESULTS

In this section, we will perform a quantitative analysis of the performance of every algorithm module. We will evaluate both the computational complexity and a set of specific metrics for every algorithmic step: Point Cloud Segmentation, Vehicle Detection, Vehicle Pose Estimation and Target Tracking.

We will evaluate the algorithms performance over a dataset acquired during the 2023 IAC @ CES competition, which took place on January 7, 2023, at the Las Vegas Motor Speedway, using the parameters described in Table VI. The dataset contains a total of 15 overtake maneuvers with opponent speeds ranging from 160 km/h to 250 km/h and EGO speeds up to 278 km/h, captured both from the overtaking and overtaken vehicle perspective.

Figure 8 shows the distribution of the computation times of the main algorithmic modules on the onboard computer during the race. The highly optimized nature of the algorithms allows for an average processing delay of 26 ms, which is half the sensor scan latency of 50 ms and, concerning end-to-end latency, lower than the state of the art for this application [21], [42].

The algorithm can actually run at 38 Hz. As the majority of the computational time is spent in segmenting the Point Clouds, due to the parallel processing of the sensors, this method will scale well on vehicles with multiple LiDAR sensors, provided that sufficient CPU cores are available.

### A. Point Cloud Segmentation

We divided the Evaluation of the Point Cloud Segmentation algorithms into its two main steps: Ground Removal and Clustering, to understand better how potential over and under-segmentation issues can affect the vehicle tracking performance.

1) Ground Removal: By proving the effectiveness of our method in removing the ground, we prove the claim regarding the high detection range and robustness of our algorithm, as effective obstacle detection requires a reliable separation of obstacles from the ground.

The Ground Removal problem can be treated as a binary classification problem, with the positive class being the non-ground objects (cars, walls, obstacles) while the negative class being ground and noise/outliers. Therefore, its performance can be evaluated by means of the classical True Positive Rate (TPR), Positive Perceived Value (PPV), and F1-score metrics.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b06b4130-43d8-4407-a073-fe96d8f8adf8/markdown_0/imgs/img_in_chart_box_121_110_575_514.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A46Z%2F-1%2F%2F68d465ebed530e60b35125a8835acae45e1dbfb9bcaf61eab9240af7ddb33bff" alt="Image" width="37%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 8. Violin plots of the computation time of the main modules of the proposed method on the 2.0 GHz Intel Xeon CPU of the onboard computer. The distributions are cropped to the 5th and 95th percentile, with the 25th and 75th percentiles and the mean value highlighted.</div> </div>


In order to determine these metrics for every LiDAR Point Cloud in the dataset, we developed an offline automatic labeling algorithm consisting of three steps: first, the normal version of every scan point is computed by fitting a plane to its neighbors. Then, by imposing a suitable threshold over the components of the normal version, all the points with a vertical normal are labeled as ground, and all the other points (belonging to the wall, opponents, or objects outside the track area) are labeled as obstacle. The final step consists in using the K-NN labeling algorithm to remove any outliers.

With the reconstructed ground truth, it is possible to compute the TPR, PPV, and F1-score metrics for every LiDAR scan and to compare the proposed algorithm with a benchmark. As a benchmark, we chose the 2.5D binary grid approach presented in [47], representing a widely accepted solution to the Ground Removal problem which has already been applied in the IAC context by [20].

Figure 9 shows the experimental results of the two algorithms over the labeled dataset. The distribution of the metrics shows how the proposed algorithm outperforms the benchmark in terms of TPR, where the benchmark shows low consistency in its performance while it underperforms in terms of PPV. The F1-score of the proposed method is larger and more consistent, showing superior overall performance. For this application, a higher PPV translates into robustness to outliers, as fewer non-ground points are labeled as ground, while TPR directly correlates with the vehicle detection performance, both in terms of detection range and number of points per vehicle.

Since this is only the first step of the LiDAR-based perception pipeline, we considered it favorable to have a better performing algorithm as there are several additional processing steps that can remove the false positives, but there is no chance of regaining a false negative. Therefore, it is possible to conclude that the proposed method outperforms the benchmark for the metrics relevant to our application.

2) Clustering: The main metric used in literature to evaluate the performance of a clustering/segmentation algorithm is the Intersection over Union (IoU) metric. However, the computation of this metric requires a segmented ground truth, which requires significant effort to build manually and proved to be very sensitive to offsets and delays when trying to build it automatically. Furthermore, for this application, we are only interested in the correct segmentation of the points belonging to the opponent vehicles.

Therefore, we preferred to use a pair of custom metrics to evaluate the performance of the proposed algorithm for this scenario. We then compared the proposed Range-Image approach [26] with the standard Euclidean Clustering algorithm from [23].

The first evaluation metric consists of the number of clusters containing at least one vehicle point. As only one vehicle is present in the dataset, the ideal value of this metric is one, with larger values showing a tendency of the algorithm to over-segment the car. The second metric, on the other hand, aims to measure the tendency of the algorithm to under-segment the opponent vehicle Point Cloud as it counts the clusters which contain both points belonging to the vehicle and to other non-vehicle obstacles (usually, the wall).

As the vehicles competing in the IAC are equipped with the same sensor configuration, the timestamped RTK GNSS pose history of the two cars can be synchronized and smoothed in order to build a ground truth of the x and y position and differential velocity v and heading  $ \psi $. These two metrics are easy to compute: by using the synchronized EGO and Opponent RTK GNSS poses, is it possible to compute the amount of clusters falling entirely inside an over-dimensioned opponent bounding box, which works under the assumption of a single opponent being present on the track, with no other obstacles. On the other hand, a mixed cluster will have dimensions much larger than a single car.

Figure 10 shows the results of this comparison on the labeled dataset by varying the main parameters of the two algorithms: the  $ th_{seg} $ angle for the Range Image Segmentation and the maximum distance for the Euclidean Clustering. It

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b06b4130-43d8-4407-a073-fe96d8f8adf8/markdown_0/imgs/img_in_chart_box_650_1106_1081_1379.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A46Z%2F-1%2F%2F89a774e7733e7aca0535d858d5ab3321ebfdf593d3b2f839b2c47b8ee5f48f50" alt="Image" width="35%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 9. Violin plots of the True Positive Rate (TPR), Positive Perceived Value (PPV), and F1-score scan-wise performance of the proposed ground segmentation method against the benchmark presented in [47] over the reconstructed ground truth. The distributions are cropped to the 95th percentile, with the 25th and 75th percentiles and the mean value highlighted.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b06b4130-43d8-4407-a073-fe96d8f8adf8/markdown_1/imgs/img_in_chart_box_130_108_564_322.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A48Z%2F-1%2F%2F232b5fc76063802627ee470123ca2111f97b220496d7926f8644ef62a6d8f44a" alt="Image" width="35%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b06b4130-43d8-4407-a073-fe96d8f8adf8/markdown_1/imgs/img_in_chart_box_128_337_564_487.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A48Z%2F-1%2F%2Fab2bf17a8ce9ae9eac64a82a83d683334ac8042c8b4284f7b29bf2de613f0d0b" alt="Image" width="35%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 10. Clustering performance computed as average and standard deviations of the number of clusters (a) per car and mixed car-wall clusters (b) for different parameter values, respectively the  $ \beta $ angle for [26] and the cluster size for [23]. The two curves are shifted horizontally for visualization purposes.</div> </div>


shows a clear tendency of the proposed algorithm to over-segment the car, however with considerably fewer mixed car-wall clusters.

For this application, over-segmenting the car usually results in having the Point Cloud of a wheel separated from the main vehicle body, which does not significantly impact the classification and opponent pose estimation tasks. On the other hand, a mixed car-wall cluster would be a disaster for the rest of the perception pipeline. Additionally, the dataset was acquired with a single opponent vehicle, but if we consider the scenario of two racecars engaging in a close overtaking maneuver, the danger of under-segmenting the two vehicles into a single cluster would severely impair the tracking performance.

For these reasons, we decided to empoly the Range Image clustering algorithm, with the  $ th_{seg} $ angle set to  $ 2.5^{\circ} $ in order to reduce the effects of over-segmentation as much as possible without risking having mixed car-wall clusters.

### B. Vehicle Detection

The green curve in Figure 11 shows the probability of true and false detections as a function of the opponent (or false positive) longitudinal distance. The opponent detection probability is obtained by dividing the number of opponent detections at a given range by the total number of LiDAR scans where the opponent is at the same distance (computed using the GNSS ground truth) whether it is detected or not. It is effectively a True Positive Rate (TPR). The false positives are normalized over the total number of LiDAR scans.

Figure 11 shows how this algorithm manages to achieve a detection range up to 90m ahead and 85m behind the EGO vehicle, with a 50% detection probability of a vehicle located 80m ahead in the traveling direction. The effect of the blind spot in the LiDAR Point Cloud due to the rear wing of the EGO vehicle can clearly be seen from the great reduction in the opponent detection probability in the [-20m,-40m] range.

The false positive detections are mostly located in the [-70m,-90m] range, as they are mainly caused to the EGO heading estimation error. Due to the effect of the car sideslip, the estimated vehicle heading coming from the differential GNSS position history tends to be closer to the vehicle's course direction than to the effective heading of the vehicle body. This leads to objects outside the racetrack limits being projected inside the track area when consulting the map.

### C. Vehicle Pose Estimation and Tracking

Figure 12 shows the distribution of the LiDAR measures and the Tracking state estimation error computed with respect to the GNSS ground truth. For this analysis, we used only the measurements and tracks corresponding to the opponent, obtained by segmenting the dataset using the Cartesian distance from the ground truth.

The analysis of the errors in the local reference frame is particularly helpful in identifying the bias present in the relative $x$ direction, which can be due to sensor miscalibration. On the other hand, the non-zero mean, high variance, and asymmetrical distribution in the relative $y$ error show an overrepresentation of the negative values. This can be explained once again by the effect of the significant vehicle side slip on the EGO state estimation error. Since the dataset is composed only of left turns, a positive sideslip would result in an artificial bias of the measurements towards the right of the EGO vehicle (negative $y$). The $\psi$ error shows a clear bivariate distribution, coming from the system switching between the outputs of the double estimators.

The effect of the tracking filters helps to reduce the bias in the local reference frame. The chosen tuning of the EKF covariance matrices aims to have a smooth estimation of the opponent velocity to allow for precise planning of overtaking and avoidance maneuvers by sacrificing the precision on the

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b06b4130-43d8-4407-a073-fe96d8f8adf8/markdown_1/imgs/img_in_chart_box_639_974_1110_1380.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A49Z%2F-1%2F%2F1c91c5269cbabaa57ba66a5a76678faba3c3cbef1bfa4f48d2ef96ba7886f87c" alt="Image" width="38%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 11. LiDAR detection (green) and Target Tracking confirmation (purple) probabilities given the real opponent longitudinal distance from the EGO vehicle. The top graph represents the probability of detecting an opponent at a certain distance (higher=better). The bottom graph represents the probability of a detection at a certain distance being a False Positive (lower=better).</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b06b4130-43d8-4407-a073-fe96d8f8adf8/markdown_2/imgs/img_in_chart_box_200_131_1032_467.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-17T14%3A21%3A50Z%2F-1%2F%2Fa7f20a3895363fae72fcc3a84ecb0039876eac76697a55d291bf8abbce469663" alt="Image" width="67%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 12. Violin plots of the raw measurements and tracked state estimation error computed with respect to the ground truth. The x and y position errors are computed both on the map and the vehicle-fixed reference frames. The distributions are cropped to the 95th percentile, with the 25th and 75th percentiles and the mean value highlighted.</div> </div>


x and y position while maintaining the estimation error for all the states under acceptable levels.

Another beneficial aspect of the target tracking algorithm is the added persistence and outlier detection capabilities obtained by the track management logic. The purple Curve in Figure 11 shows how the track management logic allows to continue estimating the opponent state even in the regions subject to a low detection probability due to the LiDAR blind spots. At the same time, the probability of tracking a false positive detection is greatly reduced.

## VI. CONCLUSION

This work presented an online LiDAR-based vehicle detection and tracking algorithm for autonomous racing, composed of multiple modules solving many crucial problems of vehicle detection from Point Cloud data. The algorithm underwent extensive field testing during the 2023 IAC @ CES competition, which involved head-to-head overtaking maneuvers at the Las Vegas Motor Speedway. Notably, this algorithm allowed team PoliMOVE to successfully overtake a target vehicle moving at 250 km/h.

Regarding Point Cloud Segmentation, the proposed approach outperformed a benchmark method in terms of TPR and F1-score. The ground removal algorithm effectively distinguished non-ground objects from ground and noise/outliers, leading to robust opponent vehicle detection and tracking. The clustering algorithm exhibited a tendency to over-segment opponent vehicles, which is acceptable considering the ability to filter false positives in later stages of the pipeline.

The precise clustering together with the map-based approach helps the algorithm to ensure a considerable detection range while avoiding the loss of track of the opponent vehicle. Although this came at the cost of more false positives, they were mainly due to objects outside the track surface and could be effectively filtered.

Measure Extraction and Target Tracking further improved the accuracy of the opponent vehicle's state estimation. The tracking algorithm demonstrated good performance in filtering and predicting the opponent's position, velocity, and heading. The use of an Extended Kalman Filter with appropriately tuned covariance matrices provided smooth and precise opponent velocity estimation, crucial for accurate planning of overtaking and avoidance maneuvers.

Despite challenges such as LiDAR blind spots and EGO state estimation errors affecting the performance, the proposed algorithm exhibited low tracking error and high outlier robustness. It is capable of tracking an opponent up to 80 meters ahead, with an average processing latency of 26ms.

In conclusion, our method demonstrates superior performance over current algorithmic approaches in vehicle detection and tracking. Future advancements in this domain are anticipated to be centered around Neural Network-based techniques, particularly for vehicle detection and pose estimation using raw or segmented LiDAR data. Importantly, the modular design of our algorithm allows for seamless integration of CNN components while preserving the overall architecture.

#### ACKNOWLEDGMENT

The authors would like to thank all the past and present members of the PoliMOVE Autonomous Racing Team, especially Marco Mandelli for his inputs in the conception and setup of this research, Andrea Marcer for his help in the software architecture design and online implementation, and Brandon Dixon and Robert Cole Frederick for their indispensable assistance in engineering and operating an autonomous racecar.

Furthermore, a special thanks goes to the TUM Autonomous Motorsport team for their role in the acquisition of the data used in this work and for motivating us to constantly improve our research, race after race, and to all the Indy Autonomous Challenge staff and teams for providing us with this unique research opportunity.

<div style="text-align: center;"><div style="text-align: center;">TABLE I</div> </div>


<div style="text-align: center;"><div style="text-align: center;">LIST OF PARAMETERS AND RESPECTIVE VALUES USED IN THE EXPERIMENTAL PERFORMANCE EVALUATION</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Symbol</td><td style='text-align: center; word-wrap: break-word;'>Value</td><td style='text-align: center; word-wrap: break-word;'>Description</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>n</td><td style='text-align: center; word-wrap: break-word;'>32</td><td style='text-align: center; word-wrap: break-word;'>Range/Elevation Image Rows</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>m</td><td style='text-align: center; word-wrap: break-word;'>857</td><td style='text-align: center; word-wrap: break-word;'>Range/Elevation Image Columns</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>szSG</td><td style='text-align: center; word-wrap: break-word;'>5</td><td style='text-align: center; word-wrap: break-word;'>Sawitsky-Golay filter kernel size</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>th_{gnd}</td><td style='text-align: center; word-wrap: break-word;'>20.0°</td><td style='text-align: center; word-wrap: break-word;'>Threshold for Ground Segmentation</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>th_{seg}</td><td style='text-align: center; word-wrap: break-word;'>2.5°</td><td style='text-align: center; word-wrap: break-word;'>Threshold for Range Image Clustering</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>w_{s}sz</td><td style='text-align: center; word-wrap: break-word;'>9</td><td style='text-align: center; word-wrap: break-word;'>Windows Size for NGRIR</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>th_{r}</td><td style='text-align: center; word-wrap: break-word;'>5.0m</td><td style='text-align: center; word-wrap: break-word;'>Threshold on Range for NGRIR</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>th_{mrg}</td><td style='text-align: center; word-wrap: break-word;'>1.80m</td><td style='text-align: center; word-wrap: break-word;'>Threshold for Cluster Merging</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Q_{diag}</td><td style='text-align: center; word-wrap: break-word;'>\{0.005, 0.005, 0.5, 0.005, 0.0005\}</td><td style='text-align: center; word-wrap: break-word;'>EKF state error covariance</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>R_{diag}</td><td style='text-align: center; word-wrap: break-word;'>\{5.0, 5.0\}</td><td style='text-align: center; word-wrap: break-word;'>EKF measurement error covariance</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>N</td><td style='text-align: center; word-wrap: break-word;'>20</td><td style='text-align: center; word-wrap: break-word;'>Iterations in track management</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>M_{c}</td><td style='text-align: center; word-wrap: break-word;'>6</td><td style='text-align: center; word-wrap: break-word;'>Confirmation Threshold</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>M_{e}</td><td style='text-align: center; word-wrap: break-word;'>5</td><td style='text-align: center; word-wrap: break-word;'>Elimination Threshold</td></tr></table>

## REFERENCES

[1] A. Wischnewski, M. Geisslinger, J. Betz, T. Betz, F. Fent, A. Heilmeier, L. Hermansdorfer, T. Herrmann, S. Huch, P. Karle et al., “Indy autonomous challenge-autonomous race cars at the handling limits,” in 12th International Munich Chassis Symposium 2021: chassis. tech plus. Springer, 2022, pp. 163–182.

[2] F. Leon and M. Gavrilescu, “A review of tracking and trajectory prediction methods for autonomous driving,” Mathematics, vol. 9, no. 6, p. 660, 2021.

[3] Y. Wang, Q. Mao, H. Zhu, J. Deng, Y. Zhang, J. Ji, H. Li, and Y. Zhang, "Multi-modal 3d object detection in autonomous driving: a survey," International Journal of Computer Vision, vol. 131, no. 8, pp. 2122–2152, 2023.

[4] J. Mao, S. Shi, X. Wang, and H. Li, "3d object detection for autonomous driving: A comprehensive survey," International Journal of Computer Vision, pp. 1–55, 2023.

[5] Z. Pang, Z. Li, and N. Wang, "Model-free vehicle tracking and state estimation in point cloud sequences," in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 8075–8082.

[6] S. Wang, P. Cai, L. Wang, and M. Liu, "Ditnet: End-to-end 3d object detection and track id assignment in spatio-temporal world," IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 3397–3404, 2021.

[7] Z. Fang, S. Zhou, Y. Cui, and S. Scherer, “3d-siamrpn: An end-to-end learning method for real-time 3d single object tracking using raw point cloud,” IEEE Sensors Journal, vol. 21, no. 4, pp. 4995–5011, 2020.

[8] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom, "Pointpillars: Fast encoders for object detection from point clouds," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 12697–12705.

[9] S. Shi, C. Guo, L. Jiang, Z. Wang, J. Shi, X. Wang, and H. Li, "Pv-rcnn: Point-voxel feature set abstraction for 3d object detection," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 10529–10538.

[10] S. Y. Alaba and J. E. Ball, “A survey on deep-learning-based lidar 3d object detection for autonomous driving,” Sensors, vol. 22, no. 24, p. 9577, 2022.

[11] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” The International Journal of Robotics Research, vol. 32, no. 11, pp. 1231–1237, 2013.

[12] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11621–11631.

[14] A. Kulkarni, J. Chrosniak, E. Ducote, F. Sauerbeck, A. Saba, U. Chirimar, J. Link, M. Cellina, and M. Behl, “Racecar—the dataset for high-speed autonomous racing,” in 2023 IEEE/RSJ International Conference

[13] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine et al., “Scalability in perception for autonomous driving: Waymo open dataset,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 2446–2454.

on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 11458–11463.

[15] F. Sauerbeck, S. Huch, F. Fent, P. Karle, D. Kulmer, and J. Betz, "Learn to see fast: Lessons learned from autonomous racing on how to develop perception systems," IEEE Access, 2023.

[16] A. Petrovskaya and S. Thrun, “Model based vehicle detection and tracking for autonomous urban driving,” Autonomous Robots, vol. 26, no. 2-3, pp. 123–139, 2009.

[17] A. Azim and O. Aycard, “Detection, classification and tracking of moving objects in a 3d environment,” in 2012 IEEE Intelligent Vehicles Symposium. IEEE, 2012, pp. 802–807.

[18] M. Himmelsbach, F. V. Hundelshausen, and H.-J. Wuensche, “Fast segmentation of 3d point clouds for ground vehicles,” in 2010 IEEE Intelligent Vehicles Symposium. IEEE, 2010, pp. 560–565.

[19] I. Bogoslavskyi and C. Stachniss, “Fast range image-based segmentation of sparse 3d laser scans for online operation,” in 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2016, pp. 163–169.

[20] C. Jung, A. Finazzi, H. Seong, D. Lee, S. Lee, B. Kim, G. Gang, S. Han, and D. H. Shim, “An autonomous system for head-to-head race: Design, implementation and analysis; team kaist at the indy autonomous challenge,” arXiv preprint arXiv:2303.09463, 2023.

[21] J. Betz, T. Betz, F. Fent, M. Geisslinger, A. Heilmeier, L. Hermansdorfer, T. Herrmann, S. Huch, P. Karle, M. Lienkamp et al., “Tum autonomous motorsport: An autonomous racing software for the indy autonomous challenge,” Journal of Field Robotics, vol. 40, no. 4, pp. 783–809, 2023.

[22] S. Cho, J. Kim, W. Ikram, K. Cho, Y.-S. Jeong, K. Um, and S. Sim, “Sloped terrain segmentation for autonomous drive using sparse 3d point cloud,” The Scientific World Journal, vol. 2014, 2014.

[23] R. B. Rusu and S. Cousins, “3d is here: Point cloud library (pcl),” in 2011 IEEE international conference on robotics and automation. IEEE, 2011, pp. 1–4.

[24] B. Douillard, J. Underwood, N. Kuntz, V. Vlakine, A. Quadros, P. Morton, and A. Frenkel, “On the segmentation of 3d lidar point clouds,” in 2011 IEEE International Conference on Robotics and Automation. IEEE, 2011, pp. 2798–2805.

[25] D. Zermas, I. Izzat, and N. Papanikolopoulos, "Fast segmentation of 3d point clouds: A paradigm on lidar data for autonomous vehicle applications," in 2017 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2017, pp. 5067–5073.

[26] I. Bogoslavskyi and C. Stachniss, “Efficient online segmentation for sparse 3d laser scans,” PFG–Journal of Photogrammetry, Remote Sensing and Geoinformation Science, vol. 85, pp. 41–52, 2017.

[27] A. Milioto, I. Vizzo, J. Behley, and C. Stachniss, “Rangenet++: Fast and accurate lidar semantic segmentation,” in 2019 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2019, pp. 4213–4220.

[28] X. Zhu, H. Zhou, T. Wang, F. Hong, Y. Ma, W. Li, H. Li, and D. Lin, "Cylindrical and asymmetrical 3d convolution networks for lidar segmentation," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 9939–9948.

[29] D. H. Douglas and T. K. Peucker, “Algorithms for the reduction of the number of points required to represent a digitized line or its caricature,” Cartographica: the international journal for geographic information and geovisualization, vol. 10, no. 2, pp. 112–122, 1973.

[30] Y. Ye, L. Fu, and B. Li, "Object detection and tracking using multi-layer laser for autonomous urban driving," in 2016 IEEE 19th international conference on intelligent transportation systems (ITSC). IEEE, 2016, pp. 259–264.

[31] M. Sualeh and G.-W. Kim, “Dynamic multi-lidar based multiple object detection and tracking,” Sensors, vol. 19, no. 6, p. 1474, 2019.

[32] X. Shen, S. Pendleton, and M. H. Ang, “Efficient l-shape fitting of laser scanner data for vehicle pose estimation,” in 2015 IEEE 7th International Conference on Cybernetics and Intelligent Systems (CIS) and IEEE Conference on Robotics, Automation and Mechatronics (RAM). IEEE, 2015, pp. 173–178.

[33] C. Zhao, C. Fu, J. M. Dolan, and J. Wang, "L-shape fitting-based vehicle pose estimation and tracking using 3d-lidar," IEEE Transactions on Intelligent Vehicles, vol. 6, no. 4, pp. 787–798, 2021.

[34] X. Zhang, W. Xu, C. Dong, and J. M. Dolan, “Efficient l-shape fitting for vehicle detection using laser scanners,” in 2017 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2017, pp. 54–59.

[35] D. Kim, K. Jo, M. Lee, and M. Sunwoo, "L-shape model switching-based precise motion tracking of moving vehicles using laser scanners," IEEE Transactions on Intelligent Transportation Systems, vol. 19, no. 2, pp. 598–612, 2017.

[36] S. Kraemer, C. Stiller, and M. E. Bouzouraa, "Lidar-based object tracking and shape estimation using polylines and free-space information," in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 4515–4522.

[37] S. Blackman and R. Popoli, “Design and analysis of modern tracking systems(book),” Norwood, MA: Artech House, 1999., 1999.

[38] B.-n. Vo, M. Mallick, Y. Bar-Shalom, S. Coraluppi, R. Osborne, R. Mahler, and B.-t. Vo, "Multitarget tracking," Wiley encyclopedia of electrical and electronics engineering, no. 2015, 2015.

[39] R. Pieroni, M. Corno, F. Parravicini, and S. M. Savaresi, “Design of an automated street crossing management module for a delivery robot,” Control Engineering Practice, vol. 153, p. 106095, 2024.

[40] K. Granström, S. Reuter, D. Meissner, and A. Scheel, “A multiple model phd approach to tracking of cars under an assumed rectangular shape,” in 17th International Conference on Information Fusion (FUSION). IEEE, 2014, pp. 1–8.

[41] P. Dahal, S. Mentasti, S. Arrigoni, F. Braghin, M. Matteucci, and F. Cheli, “Extended object tracking in curvilinear road coordinates for autonomous driving,” IEEE Transactions on Intelligent Vehicles, 2022.

[42] P. Karle, F. Fent, S. Huch, F. Sauerbeck, and M. Lienkamp, “Multi-modal sensor fusion and object tracking for autonomous racing,” IEEE Transactions on Intelligent Vehicles, 2023.

[43] K. Jo, M. Lee, J. Kim, and M. Sunwoo, “Tracking and behavior reasoning of moving vehicles based on roadway geometry constraints,” IEEE transactions on intelligent transportation systems, vol. 18, no. 2, pp. 460–476, 2016.

[44] dSPACE AUTERA Technical Details. Accessed: 2023-06-29. [Online]. Available: https://www.dspace.com/en/inc/home/products/hw/autera.cfm

[45] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, “Robot operating system 2: Design, architecture, and uses in the wild,” Science Robotics, vol. 7, no. 66, p. eabm6074, 2022.

[46] Luminar Hydra Specs. Accessed: 2023-06-29. [Online]. Available: https://autonomoustuff.com/-/media/Images/Hexagon/Hexagon%20Core/autonomousstuff/pdf/Luminar_Hydra_Specs

[47] M. Himmelsbach, T. Luettel, and H.-J. Wuensche, “Real-time object classification in 3d point clouds using point feature histograms,” in 2009 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2009, pp. 994–1000.

