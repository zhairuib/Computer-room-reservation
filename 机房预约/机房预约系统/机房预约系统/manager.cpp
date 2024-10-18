#include"manager.h"

//Ĭ�Ϲ���
Manager::Manager()
{

}

//�вι���
Manager::Manager(string name, string pwd)
{
	this->m_UserName = name;
	this->m_UserPassword = pwd;

	//��ʶ������ ��ȡ�������ļ��� ѧ������ʦ����Ϣ
	this->initVector();

	//��ʼ��������Ϣ
	ifstream ifs;
	ifs.open(COMPUTER_FILE, ios::in);

	computerRoom com;
	while (ifs >> com.m_ComId && ifs >> com.m_ManNum)
	{
		vCom.push_back(com);
	}
	ifs.close();



}

//�˵�
void Manager::operMenu()
{
	cout << "��ӭ����Ա:" << this->m_UserName << "��¼!" << endl;
	cout << "��������������������" << endl;
	cout << "����1.����˺š���" << endl;
	cout << "����2.�鿴�˺š���" << endl;
	cout << "����3.�鿴��������" << endl;
	cout << "����4.���ԤԼ����" << endl;
	cout << "����0.ע����¼����" << endl;
	cout << "��������������������" << endl;
	cout << "��������������������" << endl;
	cout << "��ǰѧ������:" << vStu.size() << endl;
	cout << "��ǰ��ʦ����:" << vTea.size() << endl;
	cout << "��������Ϊ:" << vCom.size() << endl;
	cout << "��������������������" << endl;
	cout << "��ѡ�����Ĳ���:" << endl;
}

//����˺�
void Manager::addPerson()
{
	cout << "����������˺ŵ�����" << endl;
	cout << "1.���ѧ��" << endl;
	cout << "2.�����ʦ" << endl;

	string fileName;//�����ļ���
	string tip;//��ʾid��
	ofstream ofs;//�ļ���������

	string errorTip;

	int select = 0;
	cin >> select;
	if (select == 1)
	{
		fileName = STUDENT_FILE;
		tip = "������ѧ��";
		errorTip = "ѧ���ظ�,��������!";
	}
	else
	{
		fileName = TEACHER_FILE;
		tip = "������ְ�����";
		errorTip = "ְ�����ظ�,��������!";
	}

	ofs.open(fileName, ios::out | ios::app);
	int id;
	string name;
	string pwd;


	cout << tip << endl;

	while (1)
	{
		cin >> id;
		bool ret = checkRepeat(id, select);
		if (ret)//���ظ�
		{
			cout << errorTip << endl;
		}
		else
		{
			break;
		}
	}

	cout << "����������" << endl;
	cin >> name;

	cout << "����������" << endl;
	cin >> pwd;

	//���ļ��������

	ofs << id << " " << name << " " << pwd << " " << endl;
	cout << "��ӳɹ�" << endl;



	system("pause");
	system("cls");

	ofs.close();

	this->initVector();
}

void printStudent(Student& s)
{
	cout << "ѧ��:" << s.m_Id << " ����:" << s.m_UserName << " ����:" << s.m_UserPassword << endl;
}

void printTeacher(Teacher& s)
{
	cout << "ְ����:" << s.m_EmpId << " ����:" << s.m_UserName << " ����:" << s.m_UserPassword << endl;
}


//�鿴�˺�
void Manager::showPerson()
{
	cout << "��ѡ��鿴����" << endl;
	cout << "1.�鿴����ѧ��" << endl;
	cout << "2.�鿴������ʦ" << endl;

	int select = 0;
	cin >> select;

	if (select == 1)
	{
		cout << "����ѧ����Ϣ����:" << endl;
		for_each(vStu.begin(), vStu.end(), printStudent);
	}
	else
	{
		cout << "����ʦ����Ϣ����:" << endl;
		for_each(vTea.begin(), vTea.end(), printTeacher);
	}

	system("pause");
	system("cls");
}

//�鿴������Ϣ
void Manager::showComputer()
{
	cout << "������Ϣ����" << endl;
	for (vector<computerRoom>::iterator it = vCom.begin(); it != vCom.end(); it++)
	{
		cout << "�������:" << it->m_ComId << " �����������:" << it->m_ManNum << endl;
	}

	system("pause");
	system("cls");
}

//���ԤԼ��¼
void Manager::cleamFile()
{
	ofstream ofs(ORDER_FILE, ios::trunc);
	ofs.close();

	cout << "��ճɹ�" << endl;
	system("pause");
	system("cls");
}

//��ʼ������
void Manager::initVector()
{
	//��ȡ��Ϣ ѧ�� ��ʦ 
	ifstream ifs;
	ifs.open(STUDENT_FILE, ios::in);
	if (!ifs.is_open())
	{
		cout << "�ļ���ȡʧ��" << endl;
		return;
	}

	vStu.clear();
	vTea.clear();

	Student s;
	while (ifs >> s.m_Id && ifs >> s.m_UserName && ifs >> s.m_UserPassword)
	{
		vStu.push_back(s);
	}

	ifs.close();

	//��ȡ��Ϣ ��ʦ
	ifs.open(TEACHER_FILE, ios::in);
	Teacher t;
	while (ifs >> t.m_EmpId && ifs >> t.m_UserName && ifs >> t.m_UserPassword)
	{
		vTea.push_back(t);
	}
	ifs.close();
}

//����ظ� 
bool Manager::checkRepeat(int id, int type)
{
	if (type == 1)
	{
		for (vector<Student>::iterator it = vStu.begin(); it != vStu.end(); it++)
		{
			if (id == it->m_Id)
			{
				return true;
			}
		}
	}
	else
	{
		for (vector<Teacher>::iterator it = vTea.begin(); it != vTea.end(); it++)
		{
			if (id == it->m_EmpId)
			{
				return true;
			}
		}
	}

	return false;
}
