#include<iostream>
#include"Identity.h"
#include<fstream>
#include<string>
#include"globalFile.h"
#include"Student.h"
#include"teacher.h"
#include"manager.h"

using namespace std;

//�����ʦ�Ӳ˵�����
void teacherMenu(Indentity*& teacher)
{
	while (1)
	{
		//�����Ӳ˵�����
		teacher->operMenu();

		Teacher* tea = (Teacher*)teacher;

		int select = 0;

		cin >> select;
		if (select == 1)//�鿴����ԤԼ
		{
			tea->showAllOrder();
		}
		else if (select == 2)//���ԤԼ
		{
			tea->validOrder();
		}
		else
		{
			delete teacher;
			cout << "ע���ɹ�" << endl;
			system("pause");
			system("cls");
			return;
		}
	}
}


//����ѧ���Ӳ˵�����
void studentMenu(Indentity*& student)
{
	while (1)
	{
		//����ѧ���Ӳ˵�
		student->operMenu();

		Student* stu = (Student*)student;

		int select = 0;

		cin >> select;//�����û�ѡ��
		
		if (select == 1)//����ԤԼ
		{
			stu->applyOrder();
		}
		else if (select == 2)//�鿴����ԤԼ
		{
			stu->showMyOrder();
		}
		else if (select == 3)//�鿴������ԤԼ
		{
			stu->showAllOrder();
		}
		else if(select == 4)//ȡ��ԤԼ
		{
			stu->cancelOrder();
		}
		else//ע����¼
		{
			delete student;
			cout << "ע���ɹ�!" << endl;

			system("pause");

			system("cls");
			return;
		}
	}
}


//�������Ա�Ӳ˵�����
void managerMenu(Indentity* &manager)
{
	while (1)
	{
		
		//���ù���Ա�Ӳ˵�
		manager->operMenu();

		//������ָ��תΪ����ָ�룬���������������Ľӿ�
		Manager* man = (Manager*)manager;

		

		int select = 0;
		cin >> select;

		if (select == 1)//����˺�
		{
			//cout << "����˺�" << endl;
			man->addPerson();
		}
		else if (select == 2)//�鿴�˺�
		{
			//cout << "�鿴�˺�" << endl;
			man->showPerson();
		}
		else if (select == 3)//�鿴����
		{
			//cout << "�鿴����" << endl;
			man->showComputer();
		}
		else if (select == 4)//���ԤԼ
		{
			//cout << "���ԤԼ" << endl;
			man->cleamFile();
		}
		else
		{
			//ע��
			delete manager;//���ٶ�������
			cout << "ע���ɹ�" << endl;
			system("pause");
			system("cls");
			return ;
		}
	}
}

//��¼����
//�ļ��� �������
void LoginIn(string fileName, int type)
{
	Indentity * person = NULL;//����ָ��ָ���������
	
	//���ļ�
	ifstream ifs;
	ifs.open(fileName, ios::in);

	//�ж��ļ��Ƿ����
	if (!ifs.is_open())
	{
		cout << "�ļ�������" << endl;
		ifs.close();
		return;
	}

	//׼�������û�����Ϣ
	int id = 0;
	string name;
	string pwd;

	//�ж����
	if (type == 1)
	{
		cout << "���������ѧ��:" << endl;	
		cin >> id;
	}
	else if (type == 2)
	{
		cout << "���������ְ����:" << endl;
		cin >> id;
	}

	cout << "�������û���:" << endl;
	cin >> name;
	cout << "����������:" << endl;
	cin >> pwd;


	if (type == 1)
	{
		//ѧ�������֤
		int fId;
		string fName;
		string fPwd;
		while (ifs >> fId && ifs >> fName && ifs >> fPwd)
		{
			//���û��������Ϣ���Աȡ�
			if (fId == id && fName == name && fPwd == pwd)
			{
				cout << "ѧ����֤��¼�ɹ�" << endl;
				system("pause");
				system("cls");

				person = new Student(id, name, pwd);
				//����ѧ����ݵ��Ӳ˵�
				studentMenu(person);
				return;
			}
		}
	}
	else if (type == 2)
	{
		//��ʦ�����֤
		int fId;
		string fName;
		string fPwd;
		while (ifs >> fId && ifs >> fName && ifs >> fPwd)
		{
			//���û��������Ϣ���Աȡ�
			if (fId == id && fName == name && fPwd == pwd)
			{
				cout << "��ʦ��֤��¼�ɹ�" << endl;
				system("pause");
				system("cls");

				person = new Teacher(id, name, pwd);
				//������ʦ��ݵ��Ӳ˵�
				teacherMenu(person);
				return;
			}
		}

	}
	else if (type == 3)
	{
		//����Ա�����֤
	
		string fName;
		string fPwd;
		while (ifs >> fName && ifs >> fPwd)
		{
			//���û��������Ϣ���Աȡ�
			if (fName == name && fPwd == pwd)
			{
				cout << "����Ա��֤��¼�ɹ�" << endl;
				system("pause");
				system("cls");

				person = new Manager(name, pwd);
				//�������Ա��ݵ��Ӳ˵�
				managerMenu(person);

				return;
			}
		}
	}

	cout << "��֤��¼ʧ�ܣ�" << endl;
	system("pause");
	system("cls");
	return;

}

int main(void)
{
	
	while (1)
	{

		cout << "����������������������������" << endl;
		cout << "  ��ӭʹ�û���ԤԼϵͳϵͳ" << endl;
		cout << "����������������������������" << endl;
		cout << "\t1.ѧ������" << endl;
		cout << "\t2.��  ʦ" << endl;
		cout << "\t3.����Ա" << endl;
		cout << "\t0.��  ��" << endl;
		cout << "����������������������������" << endl;
		cout << "����������ѡ��" << endl;



		int select = 0;


		cin >> select;
		switch (select)
		{
		case 1://ѧ��
			LoginIn(STUDENT_FILE, 1);
			break;
		case 2://��ʦ
			LoginIn(TEACHER_FILE, 2);
			break;
		case 3://����Ա
			LoginIn(ADMIN_FILE, 3);
			break;
		case 0://�˳�
			cout << "��ӭ�´�ʹ��!" << endl;
			system("pause");
			return 0;
			break;
		default:
			cout << "��������������ѡ��" << endl;
			system("pause");
			system("cls");
			break;
		}
	}

	system("pause");
	return 0;
}
