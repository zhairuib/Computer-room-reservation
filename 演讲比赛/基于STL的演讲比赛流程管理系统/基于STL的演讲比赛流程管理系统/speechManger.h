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
//����ݽ�������
class SpeechManger
{
public:
	//���캯��
	SpeechManger();

	//�˵�����
	void show_Menu();

	//�˳�ϵͳ
	void exitSystem();

	//��ʼ������������
	void initSpeech();

	//��ű�������
	int m_Index;

	//��Ա����
	//�����һ�ֱ���ѡ�ֱ������
	vector<int>v1;
	//��һ�ֽ�����ѡ�ֱ������
	vector<int>v2;
	//ʤ��ǰ����ѡ�ֱ������
	vector<int>vVictory;
	//��ű���Լ���Ӧ����ѡ������
	map<int, Speaker>m_Speaker;


	//����12��ѡ��
	void createSpeaker();

	//��ʼ���� �����������̿��ƺ���
	void startSpeech();

	//��ǩ
	void speechDraw();
	//����
	void speechContest();


	//��������
	~SpeechManger();
};

