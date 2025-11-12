cmake_minimum_required(VERSION 3.16)
project(ddos_simulation)

# Set C++ standard to 20 for NS-3 compatibility
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NS-3 paths
set(NS3_INCLUDE_DIR /home/traphan/ns-3-dev/build/include)
set(NS3_LIB_DIR /home/traphan/ns-3-dev/build/lib)

# Include directories
include_directories(${NS3_INCLUDE_DIR})

# Link directories
link_directories(${NS3_LIB_DIR})

# Find NS-3 libraries
find_library(NS3_CORE_LIB ns3-dev-core PATHS ${NS3_LIB_DIR})
find_library(NS3_NETWORK_LIB ns3-dev-network PATHS ${NS3_LIB_DIR})
find_library(NS3_INTERNET_LIB ns3-dev-internet PATHS ${NS3_LIB_DIR})
find_library(NS3_APPLICATIONS_LIB ns3-dev-applications PATHS ${NS3_LIB_DIR})
find_library(NS3_WIFI_LIB ns3-dev-wifi PATHS ${NS3_LIB_DIR})
find_library(NS3_MOBILITY_LIB ns3-dev-mobility PATHS ${NS3_LIB_DIR})
find_library(NS3_POINT_TO_POINT_LIB ns3-dev-point-to-point PATHS ${NS3_LIB_DIR})
find_library(NS3_FLOW_MONITOR_LIB ns3-dev-flow-monitor PATHS ${NS3_LIB_DIR})
find_library(NS3_NETANIM_LIB ns3-dev-netanim PATHS ${NS3_LIB_DIR})
find_library(NS3_STATS_LIB ns3-dev-stats PATHS ${NS3_LIB_DIR})


# Add executable
add_executable(ddos-simulator src/ddos-simulator.cc)

# Link with NS-3 libraries
target_link_libraries(ddos-simulator
    ${NS3_CORE_LIB}
    ${NS3_NETWORK_LIB} 
    ${NS3_INTERNET_LIB}
    ${NS3_APPLICATIONS_LIB}
    ${NS3_WIFI_LIB}
    ${NS3_MOBILITY_LIB}
    ${NS3_POINT_TO_POINT_LIB}
    ${NS3_FLOW_MONITOR_LIB}
    ${NS3_NETANIM_LIB}
    ${NS3_STATS_LIB}
    pthread rt
)