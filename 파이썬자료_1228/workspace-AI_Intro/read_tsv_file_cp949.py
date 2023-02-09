# -*- coding: utf-8 -*-

# 목표 = tsv , encoding=cp949 로 된 파일로 데이터 분석 

# 데이터 가져오는 방법
# 1. kobis 사이트 검색 
# 2. boxoffice menu 로 가서 일별조회 하여 하루치만 download
# 3. 엑셀을 열고 , 다른이름으로 저장 ( csv or txt(탭으로 분리된) ) 하기 
# 4. 저장된 txt(or csv) 를 해당 프로젝트 폴더에 이동 
# 5. 결과 = tsv file이 생성됨 , 인코딩 방식은 cp949


# 프로그래밍 순서
# 1. 파일 읽기
# 2. 확인 ( 문서의 앞줄 10줄, 뒤에서 10줄 )
# 3. 헤더 추출 
# 4. 50줄만 추출 
# 5. 50줄만 파일에 저장 

    
# 1. 파일 읽기 

import os
import csv

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'data_movie_cp949.txt')

with open( file_path, 'rt' , newline='' , encoding='cp949') as f:
    reader = csv.reader(f , delimiter='\t')
    lines=[line for line in reader]
print(lines)  

# 2 . 파일 확인
# 파일 첫 10줄 출력해보기 
for line in lines[:10]:
    print(line)

# 파일 마지막 10줄 출력해보기      
for line in lines[-10:]:
    print(line)

# 3. 헤더 생성
# movie list 헤더 찾고 생성     
header =lines[6]    
print(header)

# 4. 앞줄 50 출력
# 앞줄 50줄 출력 
movie_list = lines[8:58]

for line in movie_list:
    print(line)
    
# 5. 파일에 저장 
    
file_path_five = os.path.join(base_dir, 'data_movie_cp949_five.txt')
print(file_path_five)
with open ( file_path_five , 'wt' , newline='' ) as f: 
    writer = csv.writer(f, delimiter='\t')
    # writerow = 한줄 쓰기 / writerows = 여러줄 (list 타입)
    writer.writerow(header)
    writer.writerows(movie_list)
    
#### 2단계 
# 1. 누적관객수로 정렬 ( ranking )
# 2. 시각화 

# 50줄 따로 .txt 생성     
with open( file_path_five , 'rt' , newline='') as f:
    reader_five = csv.reader(f, delimiter ='\t')
    lines = [ line for line in reader_five]

print(lines)

# 순위와 헤더 인덱스 확인
for i, key in enumerate(lines[0]):
    print( i, key)
    
# 영화제목 과 누적관객수 만 따로 불러서 출력      
for line in lines[1:10]:
    print(line[1], line[11]) # line[1] = 영화제목 , line[11] = 누적관객수

# 누적관객수 출력 
audience = sorted(lines[1:] , key=lambda x:int(x[11].replace(',','')) , reverse=True)
for line in audience[:11]:
    print('{1:>20} {0:20}'.format(line[1] , line[11] ))
