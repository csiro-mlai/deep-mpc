import os
import platform
import sys
import multiprocessing

if __name__ == "__main__":
    import thirdparty.getBoost as getBoost
    import thirdparty.getRelic as getRelic
    import thirdparty.getSodium as getSodium
else:
    from .thirdparty import getBoost
    from .thirdparty import getRelic
    from .thirdparty import getSodium

#import thirdparty

def getParallel(args):
    par = multiprocessing.cpu_count()
    for x in args:
        if x.startswith("--par="):
            val = x.split("=",1)[1]
            par = int(val)
            if par < 1:
                par = 1
    return par


def Setup(boost, relic,sodium, install, prefix, par):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path + "/thirdparty")


    if boost:
        getBoost.getBoost(install,prefix, par)
    if relic:
        getRelic.getRelic(install,prefix, par)
    if sodium:
        getSodium.getSodium(install, prefix, par)


def Build(projectName, mainArgs, cmakeArgs,install, prefix, par):

    osStr = (platform.system())
    buildDir = ""
    config = ""
    buildType = ""
    if "--Debug" in mainArgs or "--debug" in mainArgs:
        buildType = "Debug"
    else:
        buildType = "Release"

    if osStr == "Windows":
        buildDir = "out/build/x64-{0}".format(buildType)
        config = "--config {0}".format(buildType)
    else:
        buildDir = "out/build/linux"

    cmakeArgs.append("-DCMAKE_BUILD_TYPE={0}".format(buildType))

    argStr = ""
    for a in cmakeArgs:
        argStr = argStr + " " + a

    parallel = ""
    if par != 1:
        parallel = " --parallel " + str(par)

    mkDirCmd = "mkdir -p {0}".format(buildDir); 
    CMakeCmd = "cmake -B {0} -S . {1}".format(buildDir, argStr)
    BuildCmd = "cmake --build {0} {1} {2} ".format(buildDir, config, parallel)

    
    InstallCmd = ""
    sudo = ""
    if "--sudo" in sys.argv:
        sudo = "sudo "

    if install:
        InstallCmd = sudo
        InstallCmd += "cmake --install {0} {1} ".format(buildDir, config)

        if len(prefix):
            InstallCmd += " --prefix {0} ".format(prefix)
            CMakeCmd += " -DCMAKE_PREFIX_PATH={0} ".format(prefix)
    
    print("\n\n====== build.py ("+projectName+") ========")
    print(mkDirCmd)
    print(CMakeCmd)
    print(BuildCmd)
    if len(InstallCmd):
        print(InstallCmd)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n\n")

    os.system(mkDirCmd)
    os.system(CMakeCmd)
    res = os.system(BuildCmd)
    if (res != 0):
        exit(1)

    if len(sudo) > 0:
        print("installing "+projectName+": {0}".format(InstallCmd))

    os.system(InstallCmd)


def getInstallArgs(args):
    prefix = ""
    for x in args:
        if x.startswith("--install="):
            prefix = x.split("=",1)[1]
            prefix = os.path.abspath(os.path.expanduser(prefix))
            return (True, prefix)
        if x == "--install":
            return (True, "")
    return (False, "")


def parseArgs():
    
    
    hasCmakeArgs = "--" in sys.argv
    mainArgs = []
    cmakeArgs = []

    if hasCmakeArgs:
        idx = sys.argv.index("--")
        mainArgs = sys.argv[:idx]
        cmakeArgs = sys.argv[idx+1:]

    else:
        mainArgs = sys.argv


    return (mainArgs, cmakeArgs)

def help():
    print(" --setup    \n\tfetch, build and optionally install the dependencies. \
    Must also pass --relic, --sodium and/or --boost to specify which to build. Without \
    --setup, the main library is built.")

    print(" --install \n\tInstructs the script to install whatever is currently being built to the default location.")
    print(" --install=prefix  \n\tinstall to the provided predix.")
    print(" --sudo  \n\twhen installing, use sudo. May require password.")
    print(" --par=n  \n\twhen building do use parallel  builds with n threads. default = num cores.")
    print(" --  \n\tafter the \"--\" argument, all command line args are passed to cmake")

    print("\n\nExamples:")
    print("-fetch the dependancies and dont install")
    print("     python build.py --setup --boost --relic")
    print("-fetch the dependancies and install with sudo")
    print("     python build.py --setup --boost --relic --install --sudo")
    print("-fetch the dependancies and install to a specified location")
    print("     python build.py --setup --boost --relic --install=~/my/install/dir")
    print("")
    print("-build the main library")
    print("     python build.py")
    print("-build the main library with cmake configurations")
    print("     python build.py -- -DCMAKE_BUILD_TYPE=Debug -DENABLE_SSE=ON")
    print("-build the main library and install with sudo")
    print("     python build.py --install --sudo")
    print("-build the main library and install to prefix")
    print("     python build.py --install=~/my/install/dir ")




def main(projectName):

    (mainArgs, cmake) = parseArgs()
    if "--help" in mainArgs:
        help()
        return 

    relic = ("--relic" in mainArgs)
    boost = ("--boost" in mainArgs)
    sodium = ("--sodium" in mainArgs)
    setup = ("--setup" in mainArgs)
    install, prefix = getInstallArgs(mainArgs)
    par = getParallel(mainArgs)

    if(setup):
        Setup(boost, relic,sodium,install, prefix, par)
    else:
        Build(projectName, mainArgs, cmake,install, prefix, par)

if __name__ == "__main__":

    main("cryptoTools")
