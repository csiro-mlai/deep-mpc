/*
 * ValueInterface.cpp
 *
 */

#include "bigint.h"
#include "ValueInterface.h"

#include <sys/stat.h>

void ValueInterface::check_setup(const string& directory)
{
    struct stat sb;
    if (stat(directory.c_str(), &sb) != 0)
        throw setup_error(directory + " does not exist");
    if (not (sb.st_mode & S_IFDIR))
        throw setup_error(directory + " is not a directory");
}
