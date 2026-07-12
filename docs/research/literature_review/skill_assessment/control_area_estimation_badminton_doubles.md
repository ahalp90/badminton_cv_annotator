<!--
provenance:
  doi: 10.1007/s11042-023-16362-1
  title: Estimation of control area in badminton doubles with pose information from top and back view drone videos
  source: unknown
  download_url: unknown
  final_host: unknown
  filed: 2026-07-12
-->


<!-- page 1 -->

# **Estimation of control area in badminton doubles with pose information from top and back view drone videos** 

#### **Ning Ding**<sup>**1**</sup> **· Kazuya Takeda**<sup>**1**</sup> **· Wenhui Jin**<sup>**2**</sup> **· Yingjiu Bei**<sup>**3**</sup> **· Keisuke Fujii**<sup>**1,4,5**</sup> 

Received: 23 May 2023 / Revised: 6 July 2023 / Accepted: 16 July 2023 / Published online: 11 August 2023 © The Author(s) 2023 

#### **Abstract** 

The application of visual tracking to the performance analysis of sports players in dynamic competitions is vital for effective coaching. In doubles matches, coordinated positioning is crucial for maintaining control of the court and minimizing opponents’ scoring opportunities. The analysis of such teamwork plays a vital role in understanding the dynamics of the game. However, previous studies have primarily focused on analyzing and assessing singles players without considering occlusion in broadcast videos. These studies have relied on discrete representations, which involve the analysis and representation of specific actions (e.g., strokes) or events that occur during the game while overlooking the meaningful spatial distribution. In this work, we present the first annotated drone dataset from top and back views in badminton doubles and propose a framework to estimate the control area probability map, which can be used to evaluate teamwork performance. We present an efficient framework of deep neural networks that enables the calculation of full probability surfaces. This framework utilizes the embedding of a Gaussian mixture map of players’ positions and employs graph convolution on their poses. In the experiment, we verify our approach by comparing various baselines and discovering the correlations between the score and control area. Additionally, we propose a practical application for assessing optimal positioning to provide instructions during a game. Our approach offers both visual and quantitative evaluations of players’ movements, thereby providing valuable insights into doubles teamwork. The dataset and related project code is available at https://github.com/Ning-D/Drone_BD_ControlArea 

**Keywords** Visual tracking · Deep learning · Drone dataset · Racket sports 

## **1 Introduction** 

Visual tracking has become an increasingly popular research area due to its diverse applications in domains such as human-computer interaction, robotics, autonomous driving, and medical imaging [3, 7, 24, 36]. Object tracking and pose estimation are two essential 

> B Ning Ding ding.ning@g.sp.m.is.nagoya-u.ac.jp 

Extended author information available on the last page of the article

<!-- page 2 -->

components of visual tracking, allowing for accurate and efficient tracking of objects in videos. However, despite significant progress in object tracking [39] and pose estimation [6], applying these technologies to sports fields remains a challenging problem due to partial occlusion, changes in viewpoint, and complex movements of athletes in a dynamic environment. 

As technology has advanced in the past few years, data collection has become more indepth and can be conducted with relative ease. Significant effort has been focused on building larger broadcast sports video datasets [8, 14]. However, broadcast videos do not show the entire pitch, only provide partial information about the game, and are mostly available to wealthy professional sports teams. To be specific, there are frequent perspective changes in broadcast video. Besides, broadcast videos also suffer from severe occlusion, especially in team sports. In soccer, drone cameras are used to capture the entire pitch in a single frame [31], providing greater adaptability and flexibility. Unfortunately, there is currently no such dataset for any racket sports. 

In this paper, we present a dataset of men’s doubles badminton matches, which was captured using two 4K-resolution drone cameras (Fig. 1a). Two drones filmed the court from the top view and back view, respectively, capturing the entire court without being affected by the occlusion problem. Based on this dataset, we propose a novel visual analysis method, as illustrated in Fig. 1b. 

Visual analytics, including heatmaps, have become prevalent in various sports, especially in team sports like soccer [4, 12]. Advanced methods are being developed to estimate probabilities of actions, such as passes, and other performance metrics based on spatiotemporal data. For example, SoccerMap [11] estimates probability surfaces of potential passes from high-frequency spatiotemporal data in team sports like soccer. However, there has been little exploration of fine-grained spatial analysis in racket sports (for more information, please refer to the next section). In racket sports, most previous studies have focused on analyzing and assessing singles players in broadcast videos and the discrete representations such as 



![Fig. 1](figures/control_area_estimation_badminton_doubles/page02_img0.png)

**Fig. 1** Contribution of our work. (a) Overview of our open-source dataset. (b) Control area estimation as a new visual analysis method

*[Illustrative overview figure: dataset sample frames and an example control-area heatmap; no data values.]*

<!-- page 3 -->

stroke analysis (e.g., [9, 13, 17]). However, these studies tend to overlook meaningful spatial distributions. 

Therefore, the objective of this study is to address these limitations by focusing on quantifying spatial value occupation and providing a quantitative position evaluation metric in badminton doubles. To achieve this, we leverage a self-collected drone dataset to estimate the control area probability map, as depicted in Fig. 1. Our proposed approach employs a two-stream network architecture (Fig. 1b) that combines positional information from a top view with pose information from a back view. By utilizing a 3-layer U-Net model, our method accurately predicts the probability map, allowing for an effective evaluation of the team’s control area. 

The contributions of this paper are as follows: 

- We build and share a men’s doubles badminton drone dataset, which includes annotations of bounding boxes, shuttlecock locations (Hit/Drop), and poses. 

- We present an efficient framework of deep neural networks that enables the calculation of full probability surfaces, which utilizes the embedding of a Gaussian mixture map of players’ positions and graph convolution of their poses. We validate the effectiveness of this fine-grained analysis of game situations in badminton. 

- We also propose a practical application for assessing optimal positioning in badminton that can increase the probability of a successful shot. 

This paper is organized as follows. First, in Section 2, We provide an overview of the related work. Next, we describe the badminton drone dataset in Section 3 and explain our approach in Section 4. Then, we present the experimental results in Section 5 and conclude this paper in Section 6. 

## **2 Related work** 

### **2.1 Visual analytics in team sports** 

Visual analytics has developed rapidly and is widely used in various sports [10, 25]. In soccer, the Time-to-intercept method is used to calculate a pitch control function that quantifies and visualizes the regions of the pitch controlled by each team [32]. Another approach uses a generative model for multi-agent trajectory data and visualizes the predicted trajectory of players in both soccer and basketball [38]. 

Other game analysis methods use non-sports-specific visualizations. One of these visualizations is the heatmap, which visualizes the most frequent locations of game events by density. In the study conducted by Fernandez et al. [11], a deep learning architecture was proposed to estimate the probability surfaces of potential passes in soccer, which is closely related to our research. However, a crucial aspect that was overlooked in their study is the impact of player poses on the probability maps. SnapShot [27] introduced a specific type of heatmap called radial heatmap to display shot data in ice hockey, while CourtVision [15] quantifies and visualizes the shooting range of players in basketball. Another commonly used visualization method is the flow graph, where the nodes’ size shows each player’s role, and the links show the connections between them [26]. Besides, the glyph-based visualization method has also been applied to sports. For example, MatchPad [22] adopts a glyph-based visual design to analyze the performances of players during rugby games. All these works demonstrate the need, impact, and potential of visual analytics in sports.

<!-- page 4 -->

### **2.2 Visual analytics in racket sports** 

In racket sports, most previous studies focused on analyzing and assessing singles players and visualizing discrete representations (e.g., stroke). In tennis, CourtTime [29] introduced a novel visual metaphor to facilitate pattern detection, while Tennivis [28] visualized statistical data, such as score and service information. In table tennis, Tac-miner [35] developed a visual analytics system to facilitate simulative analysis based on the Markov Chain, while iTTVis [37] used a matrix to reveal the relationship among multiple attributes within strokes. Tac-Simur [34] provided an interactive exploration of diverse tactical simulation tasks and visually explained the simulation results. In badminton, TIVEE [5] studied tactic analysis in a 3D environment and proposed immersive visual analytics. Recently, Haq et al. [16] utilized player tracking to visualize position statistics using heatmaps. However, this approach suffers from drawbacks such as information loss during data aggregation and reliance on implicit assumptions about data distribution. In contrast, our approach offers higher precision, detailed representations, and the capability to handle dynamic data for temporal analysis. 

### **2.3 Racket sports datasets** 

Broadcast videos have been widely used as the dataset in racket sports. Recently, significant efforts have been made to build larger broadcast sports video datasets. In badminton, YouTube videos from the Badminton World Federation have been used in academic data analysis studies [1, 9]. However, broadcast videos often suffer from frequent perspective changes and occlusion issues [1, 13]. 

Other works have collected and built their dataset based on specific task requirements, providing an optimal perspective for analysis. For instance, in tennis, TTNet [33] introduced the OpenTTGames dataset for game events detection and semantic segmentation tasks. In table tennis, Blank et al. [2] attached inertial sensors to rackets to collect stroke data, while Kulkarni et al. [21] positioned their cameras and vibration sensors to capture the most optimum view for detailed stroke analysis. 

## **3 Dataset** 

### **3.1 Video collection** 

Wecollectedourdatafrom2-vs-2men’sdoublesbadmintongamesplayedamongmembersof a college badminton club. Prior to data collection, we obtained approval from Anhui Normal University’s ethics committee (approval number [AHNU-ET2022042]) on 14th April 2022, and we conducted the study in compliance with the principles of the Declaration of Helsinki. All participants provided signed informed consent. To capture the entire badminton court, we used two DJI Air 2S drones (Da-Jiang Innovations Science and Technology Co., Ltd., China) that provided top and back views. The video resolution was 4K (3,840 × 2,160 pixels), and the frame rate was 30 fps. Our raw video data included 39 games, involving 14 pairs, 11 players, and a total of 1347 rallies. 

It is important to highlight that publicly available badminton datasets primarily consist of broadcast videos sourced from the Badminton World Federation (BWF) channel on YouTube. However, these videos typically only offer back-view footage and lack the crucial top-view videos required for our proposed method.

<!-- page 5 -->

### **3.2 Data annotation and structure** 

This section outlines our approach for efficiently annotating bounding boxes and shuttlecock locations in drone videos (Fig. 1a). Manual annotation is a time-consuming process that can take several hours to annotate a single rally. Therefore, we utilized well-established computer vision techniques to shorten the process as shown in Fig. 2. 

The dataset is structured such that each rally is accompanied by two annotation files in a simple comma-separated value (CSV) format. For shuttlecock detection, each line of the CSV file contains five values: frame number, visibility, x-coordinate, y-coordinate, and status. The status of the shuttlecock in each frame was annotated with one of five types: Frying, Hit, Fault, Drop, and Misjudge. For players, we also provide corresponding bounding box coordinates in CSV format. 

To track players and the shuttlecock in the raw video data, we first use homography transformation to eliminate the offset problem caused by the drone’s perspective. We segmented each game into several rallies, and for each rally, we estimated the XY-coordinate values for the locations of the four players using tracking, specifically ByteTrack [39], a popular high-precision multi-object detection and tracking system. We detected the shuttlecock location using TrackNet [18], an object tracking network that has been proven to exhibit decent tracking capability in games that involve small, high-speed balls such as shuttlecocks. We adjusted any outliers using a simple labeling tool with a MATLAB GUI, modifying the code originally created and utilized in the study of TrackeNet [18] to fit our study’s requirements. Moreover, we utilized direct linear transformation (DLT) to re-identify players’ IDs and manually corrected any discrepancies. By combining detection results with manual adjustments, we were able to significantly expedite the annotation process. 



![Fig. 2](figures/control_area_estimation_badminton_doubles/page05_img1.png)

**Fig. 2** Dataset annotation workflow for top-view and back-view drone videos

*[Workflow schematic; process described in the surrounding text; no data values.]*

<!-- page 6 -->

## **4 Estimations of control area** 

We propose a framework for estimating the probability map of the control area. To achieve this, we construct a neural network that learns the relationship between tracking/event data and the map in a data-driven manner. In this section, we will describe the architecture of the neural network and the learning method. 

### **4.1 Model architecture** 

The model architecture consists of a two-stream network. The first stream is based on the top view and captures information about the location and velocity of players, while the second stream is based on the back view and captures information about the pose of players. The outputs of the two streams are then combined to serve as the input of a 3-layer U-Net model, which predicts the control probability map. 

A 3-layer U-Net network was constructed to estimate the full probability map of the control area, considering that the input image size is 112 × 56. Here, the term ‘3-layer’ refers to the fact that both the downsampling layer (Max pooling) and the upsampling layer (Up-convolution) in the network consist of three layers each, as shown in Fig. 3. U-Net [30] is a convolutional network architecture specifically designed for fast and accurate image segmentation. We expect that this network can be effectively utilized in our work to segment and identify areas that are controllable or not. The network takes as input: (1) a Gaussian mixture probability map centered on the location of 2 players (same team) who receive the shuttlecock, (2) the X-velocity and Y-velocity of 2 players (same team) who receive the shuttlecock, and (3) the poses of 2 players (same team) who receive the shuttlecock. The location where the player hits the shuttlecock or where the shuttlecock lands (drop) is used as the target location to obtain the control area probability map. 

### **4.2 Learning** 

Our proposed loss function _L_ consists of a Focal loss and a constraint on spatial continuity, denoted as follows: 

$$L = L_f + \mu L_c \tag{1}$$

Here, _μ_ represents the weight for balancing the two constraints, where we set _μ_ = 0 _._ 03. As the objective function of our model, we use Focal Loss [23] to address the class imbalance problem in our dataset, defined as: 

$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{otherwise,} \end{cases} \tag{2}$$

$$FL(p_t) = -\alpha (1 - p_t)^{\gamma} \log(p_t), \tag{3}$$

where _pt_ is the estimated probability of the model. We set _α_ = 0 _._ 8 and _γ_ = 3. The focal loss _L f_ can then be written as: 

$$L_f = FL\left(y_{loc_k}, f(x_k; \theta)_{loc_k}\right), \tag{4}$$

where _xk_ is the game state at time _k_ , _lock_ represents the location where the player hit the shuttlecock or where the shuttlecock dropped at time _k_ , and _ylock_ represents the ground truth control probability at time _k_ .

<!-- page 7 -->

![Fig. 3](figures/control_area_estimation_badminton_doubles/page07_img2.png)

**Fig. 3** Detailed architecture diagram of the proposed model, showing the two-stream network and the 3-layer U-Net model used to generate the control area probability map 

*[Architecture schematic. Key details legible in the diagram: top view is preprocessed (frame alignment) then ByteTrack, giving a 112 x 56 x 3 input (Gaussian mixture map of the receiving team's player locations, plus x-velocity and y-velocity channels). Back view goes through MMPose then a GCN with shared weights across the two players, giving 112 x 56 x 48 pose features (12 keypoints with 4 features each). Concatenated 112 x 56 x 51 input feeds the 3-layer U-Net (channels 51-32-32, then 64-64, 128-128, 256 at the bottleneck; up path symmetric; conv 3x3 ReLU, max pool 2x2, up-conv 2x2, final conv 1x1 sigmoid) producing the control area probability map, trained with single pixel loss L_f plus consistency loss L_c.]*

We also use an additional constraint _L c_ , similar to Kim et al. [20]. This loss term is used to encourage spatial smoothness and continuity in the estimated control probabilities. The spatial continuity loss is defined as follows: 

$$L_c = \sum_{i=1}^{W-1} \sum_{j=1}^{H-1} \left\| v_{i+1,j} - v_{i,j} \right\|_1 + \left\| v_{i,j+1} - v_{i,j} \right\|_1, \tag{5}$$

where _W_ and _H_ represent the width and height of the image, respectively. _vi, j_ denotes the control probability of the pixel at coordinates _(i, j)_ . The loss is calculated by summing the _L_ 1-norm differences between adjacent pixels in both horizontal and vertical directions. Minimizing this loss term encourages the network to generate spatially continuous control probabilities with smooth transitions and discourages the presence of complex patterns or abrupt changes in the probability map. At the same time, it can also mitigate the impact of data amount on the results to some extent.

<!-- page 8 -->

The network is trained using the Adam optimizer with a learning rate of 10<sup>−6</sup> , 30 epochs, and batch sizes of 16. For the learning, we augmented our dataset with a horizontal flip. We have 12,658 hit samples and 796 drop samples. For the training phase, we used a ratio of 0.8 hit samples and 0.5 drop samples, and the rest for testing. 

### **4.3 Optimal positioning** 

Our model can estimate the control area and evaluate the players in badminton doubles. However, determining the optimal positioning and movement strategy for players, which is a critical aspect of successful performance, is currently unknown. In this study, we propose an approach to identify optimal positioning strategies for doubles badminton players based on data-driven analysis. 

We define the control probability function _Pc(x, y)_ as a function that measures the probability of successfully controlling the shuttlecock when the receiver is located at grid location _(x, y)_ , and the teammate is located at the same position as in the actual play. The notation _Pc(x, y) >_ = _p_ denotes the grid location where the control probability is greater than or equal to _p_ . To reduce the impact of randomness or variability, we first select the set of _n_ nearest grid locations to the actual play position of the receiver that are within _Pc(x, y) >_ = _p_ . However, we note that these _n_ nearest grid locations may be separated and not closely located together as shown in Fig. 4a. To determine the optimal location, we use unsupervised clustering, specifically the hierarchy clustering [19], to group these _n_ locations into clusters and identify the largest cluster _Cmax_ among them. We then calculate the average value of all locations in _Cmax_ to obtain the recommended position for the receiving player that increases the probability of controlling the shuttlecock to _p_ as shown in Fig. 4b. Equation (6) shows that the recommended position for the receiver is obtained by calculating the average value of all grid locations in the largest cluster _Cmax_ , which is denoted by _(xrec, yrec)_ . The formula is defined as: 

$$(x_{rec}, y_{rec}) = \frac{1}{n(C_{\max})} \sum_{(x,y) \in C_{\max}} (x, y) \tag{6}$$

Here, _n(Cmax )_ denotes the number of grid locations in the largest cluster. Overall, our approach provides a data-driven and actionable recommendation for players and coaches to follow in practice and competition. 

## **5 Results** 

In this section, we first verified the control area estimation model by visualizing the estimated control area and evaluating the accuracy. Second, we examined the practical usefulness of our approach by investigating the relationship between the control area and the score. 

### **5.1 Control area estimation** 

First, we visualized the control area when a team reacts to an incoming shuttlecock after the shuttlecock crosses the net. The changes in the control area can help to understand the reasons for losing points. Figure 5 presents the changes in the control area probability map of the receivers (on both sides) during a catch in a rally. Receivers can hit (receive) the shuttlecock in cases shown in Fig. 5a, b, and c. In the last case shown in Fig. 5d, the left-side receivers

<!-- page 9 -->

![Fig. 4](figures/control_area_estimation_badminton_doubles/page09_img3.png)

**Fig. 4** (a) This is an example of five optimal position candidates (magenta rectangles) within the control probability of _Pc(x, y) >_ = 0 _._ 95, which are separated into two clusters, illustrating the need for unsupervised clustering. (b) The recommended optimal position in magenta circle is the average value of all locations in the largest cluster 

*[Illustrative example heatmaps; no data values beyond the caption.]*

failed to catch the shuttlecock. We did not make any assumptions about the shape of the distribution in the control area (i.e., learned from data). Additionally, we observed that the model may learn the player’s speed for estimating the occupied spaces of the control area. 

### **5.2 Verification of our method** 

In this study, we evaluated the performance of our model and examined the impact of players’ velocity (− Players’ velocity), players’ pose (− Pose), and computed bounding box (Bbox) height and width from top-view (− Pose + Bbox (top)) on the estimation performance. 

The last one replaced the players’ poses in the full model with height and width measurements of bounding boxes (bboxes) as input to examine the effectiveness of the pose information. To evaluate the model, we computed the _L_ 1 classification loss between the ground truth and estimated positions for hit/drop samples. As presented in Table 1, a model trained with the full components (players’ velocity and pose) achieved the best performance for both control (hit) and non-control (drop) samples, indicating that both the players’ velocity and pose information contributed to accurately estimating the control area probability

<!-- page 10 -->

![Fig. 5](figures/control_area_estimation_badminton_doubles/page10_img4.png)

**Fig. 5** Visualization of control area changes of the receivers during catches in a rally (time passes from top to bottom). Orange and lime circles represent players’ locations and the shuttlecock location, respectively, while the arrows indicate the velocity vector for each player. A darker shade of red denotes a higher probability of control 

*[Qualitative visualization: 4 example sequences (a-d) of 6 frames each; (a-c) end in successful receives, (d) in a failed catch; no data values.]*

map. Notably, Bbox (top) was not included in the full components. We also found that using the back view poses produced more accurate results compared to using the bboxes from the top view. In the model verification stage, we achieved an overall _L_ 1 classification loss of 0.094 for the test samples with hit and drop shuttlecocks. Specifically, the _L_ 1 classification loss for hit and drop samples was 0.085 and 0.238, respectively. 

### **5.3 Relationship with the score** 

### **5.3.1 Control area in the full field** 

We defined the full field as half of the input map size (56 × 56) on the side of the receiving team, as shown in Fig. 6a. First, we examined the relationship between the score and the size 

**Table 1** Comparison of _L_ 1 classification loss for each input feature by eliminating the different input features from the total input 

|Error|−Players’ velocity|−Pose|−Pose+Bbox (top)|Full|
|---|---|---|---|---|
|Control (hit)|0.118|0.100|0.092|**0.085**|
|Non-control (drop)|0.244|0.294|0.252|**0.238**|
|All|0.125|0.110|0.101|**0.094**|

<!-- page 11 -->

![Fig. 6](figures/control_area_estimation_badminton_doubles/page11_img5.png)

**Fig. 6** (a) Definition of the full field. (b) Correlation between the score and the size of the control area 

*[Data-bearing scatter, transcribed from the render. (a) The full field is the receiving team's half, 56 x 56 grid; a control-area heatmap is overlaid. (b) Score (x, range 6 to 20) vs control-area size (y, range about 655 to 705), one point per game: rho = 0.060, p = 0.603 (no correlation).]*

of the control area on the full field. Figure 6b indicates that there is no correlation between the score and the size of the control area ( _p >_ 0 _._ 05). We speculate that the size of the control area may be related to the velocity of two players rather than the team’s performance (defense capabilities). Therefore, in the next section, we analyzed the size of the control area in the primary area instead of the full field. 

### **5.3.2 Control area in primary field** 

Next, we defined the primary field as the field where the shuttlecock is located, which has a size of one-quarter of the input map size (56 × 28). In Fig. 7, we analyzed the control area in the primary field, which depends on the shuttlecock location, and the proportion of the control area to evaluate the team’s performance in coverage, i.e., defense capabilities. The score indicates the number of points scored by the team in a game, and the control area in the primary field refers to _C_ primary = control area 1 if the shuttlecock is located in the left of the field as shown in Fig. 7a. The proportion of the control area refers to $P_{\text{control area}} = \frac{\text{control area 1}}{\text{control area 1} + \text{control area 2}}$ if the shuttlecock is located in the left of the field. 

We found moderate positive monotonic correlations between the score and the control area in the primary field (Fig. 7b, _ρ_ = 0 _._ 397 ( _p <_ 0 _._ 05)) as well as between the proportion of the control area (Fig. 7d, _ρ_ = 0 _._ 434 ( _p <_ 0 _._ 05)). In addition, for each pair, we observed a strong positive monotonic correlation between the score and the control area in the primary field (Fig. 7c, _ρ_ = 0 _._ 534 ( _p <_ 0 _._ 05)), as well as between the proportion of the control area (Fig. 7e, _ρ_ = 0 _._ 613 ( _p <_ 0 _._ 05)). These results suggest that the pair with better team performance tends to have a larger control area in the primary field where the shuttlecock is located.

<!-- page 12 -->

![Fig. 7](figures/control_area_estimation_badminton_doubles/page12_img6.png)

**Fig. 7** (a) An example of how we define the primary field when the shuttlecock is located at the green circle is as follows: the primary field is the rectangular area that includes the green circle and covers one-quarter of the input map size (56 × 28). (b-e) Correlations between the score and the (b,c) control area in the primary field or (d,e) proportion of the control area for each (b,d) game or identified pair (c,e) 

*[Data-bearing scatters, transcribed from the render. Each panel prints its own rho and p on the plot. (b) control area in primary field, per game: rho = 0.397, p = 0.000. (c) control area in primary field, per pair: rho = 0.534, p = 0.049. (d) proportion of control area, per game: rho = 0.434, p = 0.000. (e) proportion of control area, per pair: rho = 0.613, p = 0.020. Panels (c) and (e) label the 14 pairs individually.]*

### **5.3.3 Length/width of the control area** 

In badminton, mastering the “doubles rotation” skill is crucial to maintain court coverage and preventing any gaps during the match. The ability to effectively cover the field is a valuable indicator of player performance. 

To measure field coverage, we analyzed the control area of each team at the moment their opponents hit the shuttlecock. This is the time when both players in the team should prepare and position themselves for the next stroke (see Fig. 8a). 

For each game and each pair, we found no correlation between the score and the length of the control area ( _p >_ 0 _._ 05), as shown in Fig. 8b and c. However, we observed a weak positive monotonic correlation between the score and the width of the control area for each game (Fig. 8d, _ρ_ = 0 _._ 249 and _p <_ 0 _._ 05), and a strong positive monotonic correlation between the score and the width of the control area for each pair (Fig. 8e, _ρ_ = 0 _._ 618 and _p <_ 0 _._ 05). These results suggest that teams with better performance tend to cover more width of the field when preparing for the next stroke. 

### **5.3.4 Aiming technique** 

In doubles badminton, players look for opportunities to hit the shuttlecock towards areas where the opposing team’s formation is relatively distant during the moment of hitting. This strategy aims to increase the likelihood of their opponents making an error. Therefore, we use the maximum distance between the position where the player aims to land the shuttlecock and the control area of opposing players across all rallies as a measure of this aiming technique,

<!-- page 13 -->

![Fig. 8](figures/control_area_estimation_badminton_doubles/page13_img7.png)

**Fig. 8** (a) Definition of the length and width of the control area of a team. (b-e) Correlations between the score and the (b,c) length or (d,e) width of the control area for each (b,d) game or identified pair (c,e) 

*[Data-bearing scatters, transcribed from the render. (b) length, per game: rho = -0.141, p = 0.217 (no correlation). (c) length, per pair: rho = -0.216, p = 0.459 (no correlation). (d) width, per game: rho = 0.249, p = 0.028. (e) width, per pair: rho = 0.618, p = 0.018.]*

denoted as _Ad_ , as shown in Fig. 9a. In cases where the player’s shot is returned by the opponent, we approximate the player’s aiming position with the opponent’s actual hitting position. 

We observed a weak positive monotonic correlation between the score and the _Ad_ for each game (Fig. 9b, _ρ_ = 0 _._ 249 and _p <_ 0 _._ 05) For each pair, we found no correlation between the score and the _Ad_ ( _p >_ 0 _._ 05), as shown in Fig. 9c. These results suggest that the aiming technique measured by _Ad_ has some impact on the overall score in doubles badminton games. However, for individual pairs, other factors such as teamwork, communication, and individual skills, may also play important roles in determining the success of a doubles badminton pair. 



![Fig. 9](figures/control_area_estimation_badminton_doubles/page13_img8.png)

**Fig. 9** (a) Definition of the aiming technique distance _Ad_ . (b) Correlation between the score and _Ad_ for each game. (c) Correlation between the score and _Ad_ for identified pair

*[Data-bearing scatters, transcribed from the render. (b) aiming distance Ad, per game: rho = 0.249, p = 0.028. (c) aiming distance Ad, per pair: rho = 0.257, p = 0.375 (no correlation).]*

<!-- page 14 -->

![Fig. 10](figures/control_area_estimation_badminton_doubles/page14_img9.png)

**Fig. 10** (a-d) The figures present the optimal positions for drop samples. The orange circles indicate the current positions of both players, while the plum and magenta circles represent the recommended positions for the receiver. By positioning the receiver in these locations, the probability of successfully controlling the shuttlecock can increase to either 0.75 or 0.95 

*[Qualitative: four control-area heatmaps (red-hot = high control probability, 0 to 1 colour bar). Orange circles are current player positions; the plum circle marks the recommended receiver position reaching Pc >= 0.75 and the magenta circle Pc >= 0.95. No new data values beyond the caption.]*

### **5.4 Assessment of optimal positioning** 

As proposed in Section 4.3, we define optimal locations as those that are near the receiver and provide a higher probability of a successful shot than the current location. To improve the chances of controlling the shuttlecock during drop shots in doubles, we recommend moving along the shortest path while maintaining the hitting pose. Specifically, we suggest fixing one player’s location and considering all possible grid locations (112 × 56) for the receiving player. 

In our case, we use our control probability model to identify the five nearest grid locations ( _n_ = 5) to the actual play position of the receiver where the probability of controlling the shuttlecock is greater than or equal to 0.75 and 0.95 ( _Pc(x, y) >_ = 0 _._ 75 and _Pc(x, y) >_ = 0 _._ 95). We then use a hierarchical clustering algorithm to group these five locations into clusters and identify the largest cluster among them. Finally, we calculate the average value of all locations in the largest cluster, which gives us a recommended position for the receiving player that increases the probability of controlling the shuttlecock to 0.75 (plum circle) and 0.95 (magenta circle), as shown in Fig. 10. 

## **6 Conclusion** 

In this study, we developed a framework to estimate the control area probability map from an input badminton drone video. We verified our approach by comparing it with various baselines and discovered valuable insights into the relationship between the score and control area. We also shared the first annotated badminton drone video dataset and provided a practical solution for evaluating optimal positioning in badminton, which can improve the likelihood of successful shots. We believe this visual tool can be extended to other racket sports. Our approach can evaluate players’ movements both visually and quantitatively, providing valuable insights into doubles teamwork for coaching, assessing, and scouting purposes.

<!-- page 15 -->

In our future work, we plan to consider more dynamic indicators to reflect the skill in Fig. 9a, and extend our framework to a variety of other racket sports, such as table tennis and tennis. We believe such visual representation of complex information can provide coaches with a deep perception of the game situation, thus providing a competitive advantage to an individual or a team. 

**Acknowledgements** This work was financially supported by JST SPRING, Grant Number JPMJSP2125, JSPS Grant Number 20H04075, JST PRESTO Grant Number JPMJPR20CA, and Scientific Research Project of Higher Education Institutions of Anhui Province of China Grant Number 2022AH052181. The author (ND) would like to take this opportunity to thank the “Interdisciplinary Frontier Next-Generation Researcher Program of the Tokai Higher Education and Research System.” The authors would also like to thank Yundong Yu for his valuable comments on this work. 

**Data Availibility** The datasets generated during and/or analyzed during the current study are available in the GitHub repository at the following link: https://github.com/Ning-D/Drone_BD_ControlArea 

## **Declarations** 

**Ethical Standards** All procedures performed in studies involving human participants were in accordance with the ethical standards of the institutional and/or national research committee and with the 1964 Helsinki Declaration and its later amendments or comparable ethical standards. The study received approval from the ethics committee at Anhui Normal University (No. AHNU-ET2022042) on April 14th, 2022, and all participants provided signed informed consent. 

##### **Conflict of Interest** The authors declare that they have no conflict of interest. 

**Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/. 

## **References** 

1. Archana, M., Kalaiselvi Geetha, M.: An efficient ball and player detection in broadcast tennis video. In: Intelligent Systems Technologies and Applications: Volume 1, pp. 427–436 (2016). Springer 

2. Blank, P., Hoßbach, J., Schuldhaus, D., Eskofier, B.M.: Sensor-based stroke detection and stroke type classification in table tennis. In: Proceedings of the 2015 ACM International Symposium on Wearable Computers, pp. 93–100 (2015) 

3. Boutteau R, Rossi R, Qin L, Merriaux P, Savatier X (2020) A vision-based system for robot localization in large industrial environments. Journal of Intelligent & Robotic Systems 99:359–370 

4. Cho H, Ryu H, Song M (2022) Pass2vec: Analyzing soccer players’ passing style using deep learning. International Journal of Sports Science & Coaching 17(2):355–365 

5. Chu X, Xie X, Ye S, Lu H, Xiao H, Yuan Z, Chen Z, Zhang H, Wu Y (2021) Tivee: Visual exploration and explanation of badminton tactics in immersive visualizations. IEEE Transactions on Visualization and Computer Graphics 28(1):118–128 

6. Contributors, M.: OpenMMLab Pose Estimation Toolbox and Benchmark. https://github.com/openmmlab/mmpose (2020) 

7. Dasgupta K, Das A, Das S, Bhattacharya U, Yogamani S (2022) Spatio-contextual deep network-based multimodal pedestrian detection for autonomous driving. IEEE Transactions on Intelligent Transportation Systems 23(9):15940–15950

<!-- page 16 -->

8. Deliege, A., Cioppa, A., Giancola, S., Seikavandi, M.J., Dueholm, J.V., Nasrollahi, K., Ghanem, B., Moeslund, T.B., Van Droogenbroeck, M.: Soccernet-v2: A dataset and benchmarks for holistic understanding of broadcast soccer videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4508–4519 (2021) 

9. Ding N, Takeda K, Fujii K (2022) Deep reinforcement learning in a racket sport for player evaluation with technical and tactical contexts. IEEE Access 10:54764–54772 

10. Du, M., Yuan, X.: A survey of competitive sports data visualization and visual analysis. Journal of Visualization 24 (2020) https://doi.org/10.1007/s12650-020-00687-2 

11. Fernández, J., Bornn, L.: Soccermap: A deep learning architecture for visually-interpretable analysis in soccer. In: Machine Learning and Knowledge Discovery in Databases. Applied Data Science and Demo Track: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14–18, 2020, Proceedings, Part V, pp. 491–506 (2021). Springer 

12. Fernandez, J., Bornn, L.: Wide open spaces: A statistical technique for measuring space creation in professional soccer. In: Sloan Sports Analytics Conference, vol. 2018 (2018) 

13. Ghosh, A., Singh, S., Jawahar, C.: Towards structured analysis of broadcast badminton videos. In: 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 296–304 (2018). IEEE 

14. Giancola, S., Ghanem, B.: Temporally-aware feature pooling for action spotting in soccer broadcasts. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4490–4499 (2021) 

15. Goldsberry, K.: Courtvision: New visual and spatial analytics for the nba. In: 2012 MIT Sloan Sports Analytics Conference, vol. 9, pp. 12–15 (2012) 

16. Haq, M.A., Tarashima, S., Tagawa, N.: Heatmap visualization and badminton player detection using convolutional neural network. In: 2022 International Electronics Symposium (IES), pp. 627–631 (2022). IEEE 

17. Hsu, T.-H., Chen, C.-H., Jut, N.P., Ik, T.-U., Peng, W.-C., Wang, Y.-S., Tseng, Y.-C., Huang, J.-L., Ching, Y.-T., Wang, C.-C., et al. Coachai: A project for microscopic badminton match data collection and tactical analysis. In: 2019 20th Asia-Pacific Network Operations and Management Symposium (APNOMS), pp. 1–4 (2019). IEEE 

18. Huang, Y.-C., Liao, I.-N., Chen, C.-H.,<sup>˙</sup> Ik, T.-U., Peng, W.-C.: Tracknet: A deep learning network for tracking high-speed and tiny objects in sports applications. In: 2019 16th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), pp. 1–8 (2019). IEEE 

19. Johnson SC (1967) Hierarchical clustering schemes. Psychometrika 32(3):241–254 

20. Kim W, Kanezaki A, Tanaka M (2020) Unsupervised learning of image segmentation based on differentiable feature clustering. IEEE Transactions on Image Processing 29:8055–8068 

21. Kulkarni,K.M.,Shenoy,S.:Tabletennisstrokerecognitionusingtwo-dimensionalhumanposeestimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4576– 4584 (2021) 

22. Legg, P.A., Chung, D.H., Parry, M.L., Jones, M.W., Long, R., Griffiths, I.W., Chen, M.: Matchpad: interactive glyph-based visualization for real-time sports performance analysis. In: Computer Graphics Forum, vol. 31, pp. 1255–1264 (2012). Wiley Online Library 

23. Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 2980–2988 (2017) 

24. Mueller, F., Bernard, F., Sotnychenko, O., Mehta, D., Sridhar, S., Casas, D., Theobalt, C.: Ganerated hands for real-time 3d hand tracking from monocular rgb. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 49–59 (2018) 

25. Perin, C., Vuillemot, R., Stolper, C.D., Stasko, J.T., Wood, J., Carpendale, S.: State of the art of sports data visualization. In: Computer Graphics Forum, vol. 37, pp. 663–686 (2018). Wiley Online Library 

26. Perin C, Vuillemot R, Fekete J-D (2013) Soccerstories: A kick-off for visual soccer analysis. IEEE transactions on visualization and computer graphics 19(12):2506–2515 

27. Pileggi H, Stolper CD, Boyle JM, Stasko JT (2012) Snapshot: Visualization to propel ice hockey analytics. IEEE Transactions on Visualization and Computer Graphics 18(12):2819–2828 

28. Polk T, Yang J, Hu Y, Zhao Y (2014) Tennivis: Visualization for tennis match analysis. IEEE transactions on visualization and computer graphics 20(12):2339–2348 

29. Polk T, Jäckle D, Häußler J, Yang J (2019) Courttime: Generating actionable insights into tennis matches using visual analytics. IEEE Transactions on Visualization and Computer Graphics 26(1):397–406 

30. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234–241 (2015). Springer

<!-- page 17 -->

31. Scott, A., Uchida, I., Onishi, M., Kameda, Y., Fukui, K., Fujii, K.: Soccertrack: A dataset and tracking algorithm for soccer with fish-eye and drone videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3569–3579 (2022) 

32. Spearman, W., Basye, A., Dick, G., Hotovy, R., Pop, P.: Physics-based modeling of pass probabilities in soccer. In: Proceeding of the 11th MIT Sloan Sports Analytics Conference (2017) 

33. Voeikov, R., Falaleev, N., Baikulov, R.: Ttnet: Real-time temporal and spatial video analysis of table tennis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 884–885 (2020) 

34. Wang J, Zhao K, Deng D, Cao A, Xie X, Zhou Z, Zhang H, Wu Y (2019) Tac-simur: Tactic-based simulative visual analytics of table tennis. IEEE transactions on visualization and computer graphics 26(1):407–417 

35. Wang J, Wu J, Cao A, Zhou Z, Zhang H, Wu Y (2021) Tac-miner: Visual tactic mining for multiple table tennis matches. IEEE Transactions on Visualization and Computer Graphics 27(6):2770–2782 

36. Wawrzyniak N, Hyla T, Popik A (2019) Vessel detection and tracking method based on video surveillance. Sensors 19(23):5230 

37. Wu Y, Lan J, Shu X, Ji C, Zhao K, Wang J, Zhang H (2017) ittvis: Interactive visualization of table tennis data. IEEE transactions on visualization and computer graphics 24(1):709–718 

38. Yeh, R.A., Schwing, A.G., Huang, J., Murphy, K.: Diverse generation for multi-agent sports games. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4610–4619 (2019) 

39. Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., Wang, X.: Bytetrack: Multiobject tracking by associating every detection box. In: Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXII, pp. 1–21 (2022). Springer 

**Publisher’s Note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and 

## **Authors and Affiliations** 

#### **Ning Ding**<sup>**1**</sup> **· Kazuya Takeda**<sup>**1**</sup> **· Wenhui Jin**<sup>**2**</sup> **· Yingjiu Bei**<sup>**3**</sup> **· Keisuke Fujii**<sup>**1,4,5**</sup> 

Kazuya Takeda kazuya.takeda@nagoya-u.jp 

Wenhui Jin jinwenhui@whit.edu.cn 

Yingjiu Bei beibei@mail.ahnu.edu.cn 

Keisuke Fujii fujii@i.nagoya-u.ac.jp 

- 1 Graduate School of Informatics, Nagoya University, Chikusa-ku, Nagoya, Aichi, Japan 

- 2 Department of Physical Education, Wuhu Institute of Technology, Wenjinxi Rd, Wuhu, Anhui, China 

- 3 School of Sports Science, Anhui Normal University, East Jiuhua Rd, Wuhu, Anhui, China 

- 4 RIKEN Center for Advanced Intelligence Project, 1-5, Yamadaoka, Suita, Osaka, Japan 

- 5 PRESTO, Japan Science and Technology Agency, Kawaguchi, Saitama, Japan