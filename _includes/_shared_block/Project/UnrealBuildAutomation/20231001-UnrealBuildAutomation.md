unreal의 uproject를 복사하여, 다른 빌드 배치를 생성하는 배치파일입니다.

1. 패스와 프로젝트 명의 변수를 설정합니다.
2. 빌드하기 위한 준비를 합니다.
3. `AutomationTool.exe`를 실행하여 언리얼엔진을 빌드합니다.

* 프로그램의 이름을 변경하기 위해 다른 .uproject를 사용합니다.
* 사용하는 플러그인으로인해 타겟이 생성되는데, 한 프로젝트에는 타겟이 하나만있어야 하지만 여러개가 있으므로 문제가 발생합니다. 목적대로 작동하게 하기 위해 Intermediate 폴더를 지웁니다.
* 사용하는 현지화 언어를 강제하기 위해 처음 프로그램실행시 참고되는 유저설정(BaseGameUserSetting)에 culture=en을 추가합니다.

```bat
:: 언리얼 엔진과 프로젝트 주소, 저장할 주소를 입력합니다.
@echo off
set EngineDirectoryPath=D:\Unreal\UE_4.27\Engine
set ProjectName=UnrealProjectName
@echo on

:: 현재 프로젝트와 폴더 경로에 
@echo off
set ProjectDictionaryPath=%cd%
set DefaultStagingDirectoryPath=%cd%
@echo on

:: 인수로 사용할 변수를 작성합니다.
@echo off
set AutomationToolPath=%EngineDirectoryPath%\Binaries\DotNET\AutomationTool.exe
set UE4ExePath=%EngineDirectoryPath%\Binaries\Win64\UE4Editor-Cmd.exe
set UProject=%ProjectName%.uproject
set ConfigPathInBuild=WindowsNoEditor\Engine\Config
set BaseGameUserSettinginBuild=%ConfigPathInBuild%\BaseGameUserSettings.ini
@echo on

:: 해당 프로그램의 Config에 마지막으로 사용한 현지화 Config가 있을 경우 초기 설정이 적용되지 않을 수 있습니다.
:Main
call :RunUAT UnrealProjectName_en
	@echo off
	set BaseGameUserSettingPath=%StagingDirectoryPath%\%BaseGameUserSettinginBuild%
	@echo on
	:: 마지막 스테이징의 BaseGameuserSetting에 culture를 추가합니다.
	echo. >> "%BaseGameUserSettingPath%"
	echo culture=en >> "%BaseGameUserSettingPath%"
call :RunUAT UnrealProjectName_ko
goto EOF

:: .uproject를 복사하고 해당 프로젝트를 이용하여 빌드합니다.
:: UAT를 이용하여 프로젝트를 빌드하고, 해당 위치에 저장합니다.
:: UAT가 종료되면 Intermediate폴더를 삭제합니다.
:: 커멘드는 unreal에서 project launch를 세팅하여 빌드할 때, 보내는 파라메터를 가져와 이용합니다.
:RunUAT <NewProjectName>
@echo off
set NewProjectName=%~1
set NewUProject=%~1.uproject
set NewUProjectpath=%ProjectDictionaryPath%\%NewUProject%
set StagingDirectoryPath=%DefaultStagingDirectoryPath%\%NewProjectName%
@echo on
@echo off
set RunCommand="%AutomationToolPath%" BuildCookRun -project="%NewUProjectpath%"
@echo on
if not exist %NewUProject% copy %UProject% %NewUProject%
if exist "Intermediate" rd /s /q "Intermediate"
%RunCommand% -noP4 -clientconfig=Shipping -serverconfig=Shipping -nocompile -nocompileeditor -installed -ue4exe="%UE4ExePath%" -utf8output -platform=Win64 -targetplatform=Win64 -build -cook -map= -unversionedcookedcontent -compressed -prereqs -stage -package -stagingdirectory="%StagingDirectoryPath%" -cmdline=" -Messaging"
if not "%ProjectName%.uproject" == "%NewUProject%" del %NewUProject%

```