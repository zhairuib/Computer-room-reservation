#pragma once
#include <iostream>
using namespace std;
#include<vector>
#include<map>
#include"speaker.h"
#include<algorithm>
#include<deque>
#include<functional>
#include<numeric>
#include<string>
//设计演讲管理类
class SpeechManger
{
public:
	//构造函数
	SpeechManger();

	//菜单函数
	void show_Menu();

	//退出系统
	void exitSystem();

	//初始化容器和属性
	void initSpeech();

	//存放比赛轮数
	int m_Index;

	//成员属性
	//保存第一轮比赛选手编号容器
	vector<int>v1;
	//第一轮晋级的选手编号容器
	vector<int>v2;
	//胜出前三名选手编号容器
	vector<int>vVictory;
	//存放编号以及对应具体选手容器
	map<int, Speaker>m_Speaker;


	//创建12名选手
	void createSpeaker();

	//开始比赛 比赛整个流程控制函数
	void startSpeech();

	//抽签
	void speechDraw();
	//比赛
	void speechContest();


	//析构函数
	~SpeechManger();
};

