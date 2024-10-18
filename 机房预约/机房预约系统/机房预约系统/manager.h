#pragma once
#include"Identity.h"
#include<iostream>
#include<string>
#include"globalFile.h"
#include<fstream>
#include<vector>
#include"Student.h"
#include"teacher.h"
#include<algorithm>
#include"computerRoom.h"
using namespace std;
class Manager :public Indentity
{
public:
	//Ĭ�Ϲ���
	Manager();

	//�вι���
	Manager(string name, string pwd);

	//�˵�
	virtual void operMenu();

	//����˺�
	void addPerson();

	//�鿴�˺�
	void showPerson();

	//�鿴������Ϣ
	void showComputer();

	//���ԤԼ��¼
	void cleamFile();

	//��ʼ������
	void initVector();

	//ѧ������
	vector<Student>vStu;

	//��ʦ����
	vector<Teacher>vTea;

	//����ظ� ѧ�š�ְ����  �������
	bool checkRepeat(int id, int type);

	//������Ϣ
	vector<computerRoom>vCom;

};

