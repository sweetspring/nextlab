Case 1 视频中的人脸检测和识别
==================================
.. highlight:: python
.. The first case. A start. Not well organized yet. 

Credit
---------------------
本例所用到的视频和图片仅用于研究，请勿用于商业用途，视频和图片所有权利归于原版权所有者。


视频获取和预处理
---------------------

对于人工智能有一个类似马太效应的传说，这个传说是，做人工智能首先要有产品，然后用这个产品去获取数据，优化产品，如果你是后来者，又没有数据，就很难再进入这个领域了。所以我们首先用这个人脸检测识别案例来说明，数据壁垒，可以想办法打破的。如何打破技术壁垒，放到其他案例中探讨。


* 想想哪里有很多人脸呢？国外有很多人脸数据库，但我们有很多电影和电视剧。想到演员，首先想到林青霞，《刀马旦》是一部很好看的电影，林青霞（Lin）着男装，而另外两位女演员叶倩文（Ye）和钟楚红（Zhong）不仔细看还有点像，我们就拿这个电影来作示例。

* 获取电影视频，以及三位女演员的若干图片 （每人至少一张）
* 视频格式转换，从 .flv 到 .mp4
* 截取视频片段，取了其中 00:25～00:30 共5分钟的片段
* 获取视频信息，了解压缩编码，帧率、尺寸等信息 : ::

	{
	    "streams": [
		{
		    "index": 0,
		    "codec_name": "h264",
		    "codec_long_name": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
		    "profile": "High",
		    "codec_type": "video",
		    "codec_time_base": "1/30",
		    "codec_tag_string": "avc1",
		    "codec_tag": "0x31637661",
		    "width": 320,
		    "height": 172,
		    "coded_width": 320,
		    "coded_height": 176,
		    "has_b_frames": 2,
		    "sample_aspect_ratio": "0:1",
		    "display_aspect_ratio": "0:1",
		    "pix_fmt": "yuv420p",
		    "level": 12,
		    "chroma_location": "left",
		    "refs": 4,
		    "is_avc": "1",
		    "nal_length_size": "4",
		    "r_frame_rate": "15/1",
		    "avg_frame_rate": "15/1",
		    "time_base": "1/15360",
		    "start_pts": 0,
		    "start_time": "0.000000",
		    "duration_ts": 4610048,
		    "duration": "300.133333",
		    "bit_rate": "103028",
		    "bits_per_raw_sample": "8",
		    "nb_frames": "4502",
		    "disposition": {
		        "default": 1,
		        "dub": 0,
		        "original": 0,
		        "comment": 0,
		        "lyrics": 0,
		        "karaoke": 0,
		        "forced": 0,
		        "hearing_impaired": 0,
		        "visual_impaired": 0,
		        "clean_effects": 0,
		        "attached_pic": 0
		    },
		    "tags": {
		        "language": "und",
		        "handler_name": "VideoHandler"
		    }
		},
		{
		    "index": 1,
		    "codec_name": "aac",
		    "codec_long_name": "AAC (Advanced Audio Coding)",
		    "profile": "LC",
		    "codec_type": "audio",
		    "codec_time_base": "1/22050",
		    "codec_tag_string": "mp4a",
		    "codec_tag": "0x6134706d",
		    "sample_fmt": "fltp",
		    "sample_rate": "22050",
		    "channels": 2,
		    "channel_layout": "stereo",
		    "bits_per_sample": 0,
		    "r_frame_rate": "0/0",
		    "avg_frame_rate": "0/0",
		    "time_base": "1/22050",
		    "start_pts": 0,
		    "start_time": "0.000000",
		    "duration_ts": 6615017,
		    "duration": "300.000771",
		    "bit_rate": "129484",
		    "max_bit_rate": "129484",
		    "nb_frames": "6460",
		    "disposition": {
		        "default": 1,
		        "dub": 0,
		        "original": 0,
		        "comment": 0,
		        "lyrics": 0,
		        "karaoke": 0,
		        "forced": 0,
		        "hearing_impaired": 0,
		        "visual_impaired": 0,
		        "clean_effects": 0,
		        "attached_pic": 0
		    },
		    "tags": {
		        "language": "und",
		        "handler_name": "SoundHandler"
		    }
		}
	    ],
	    "format": {
		"filename": "dmdpart.mp4",
		"nb_streams": 2,
		"nb_programs": 0,
		"format_name": "mov,mp4,m4a,3gp,3g2,mj2",
		"format_long_name": "QuickTime / MOV",
		"start_time": "0.000000",
		"duration": "300.134000",
		"size": "8878451",
		"bit_rate": "236652",
		"probe_score": 100,
		"tags": {
		    "major_brand": "isom",
		    "minor_version": "512",
		    "compatible_brands": "isomiso2avc1mp41",
		    "encoder": "Lavf56.40.101"
		}
	    }
	}


* 去除台标


人脸检测和识别初步
---------------------
这个视频片段有 4502 帧，我们利用已有模型 FACEA 对它进行识别，对三位女演员分别标记:林青霞（Lin）叶倩文（Ye）和钟楚红（Zhong），可见 FACEA 误检率和漏检率还是不小的。其中漏检率可以通过阈值调整减少，但与之同时会增加误检率。

请参见 dmd_face1.avi

人脸数据的获取
---------------------
我们如何来进一步获取三位女演员正确的人脸数据，也就是正确的人脸图片呢？容后再续

 
人工智能的应用
---------------------
从这个案例引申开去，字幕的翻译、视频采访中人物的模糊处理都是可以用到人工智能的。

* 字幕翻译和语音翻译可以用于国产影片出口。人工翻译比较消耗资源，如果配给一个外挂的智能翻译，加一个纠错机制，可以开拓更大的电影市场。

* 电视采访中有一些当事人需要作隐私保护，特别是儿童，要打马赛克，也是可以作人脸跟踪检测识别+图像模糊处理做到。马赛克图像处理可以参考去除台标的方法。
