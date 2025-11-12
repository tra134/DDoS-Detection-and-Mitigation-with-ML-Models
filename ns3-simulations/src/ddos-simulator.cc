#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h" // NetAnim support
#include <fstream>
#include <vector>
#include <sstream>
#include <map> 

// <<< VÔ HIỆU HÓA GNUPLOT >>>
// #include "ns3/gnuplot.h" 

// <<< THÊM THƯ VIỆN ĐỂ CHẶN IP >>>
#include <set> 

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DDoSSimulator");

class DDoSSimulator
{
public:
    DDoSSimulator(uint32_t nIotNodes, uint32_t nAttackers, double simTime);
    void Run();

private:
    uint32_t m_nIotNodes;
    uint32_t m_nAttackers;
    double m_simTime;
    NodeContainer m_iotNodes;
    NodeContainer m_baseStations;
    NodeContainer m_serverNode;
    NodeContainer m_allNodes;

    Ipv4InterfaceContainer m_staInterfaces;
    Ipv4Address m_serverIp;
    NetDeviceContainer m_apDevices; // Lưu NetDevice của AP

    std::vector<uint32_t> m_attackerIndices;
    std::vector<Ipv4Address> m_attackerIps;

    FlowMonitorHelper m_flowMonHelper;
    Ptr<FlowMonitor> m_monitor;

    std::ofstream m_statsFile; 
    std::map<FlowId, uint32_t> m_lastTxPackets; 

    // --- Biến cho Mitigation ---
    std::ofstream m_liveStatsFile; 
    std::set<Ipv4Address> m_blockedIps; 
    std::map<FlowId, FlowMonitor::FlowStats> m_lastFlowStats; 

    // --- Các hàm ---
    void CreateNodes();
    void SetupMobility();
    void SetupNetwork();
    void SetupApplications();
    void SetupDDoSAttack();
    void SetupNetAnim();
    void CollectStatistics();
    
    // <<< VÔ HIỆU HÓA GNUPLOT >>>
    // void ExportToGnuplot(); 
    
    void SetupMitigation();
    void MonitorLiveFlows();
    void CheckForBlacklistUpdates();
    
    // Chữ ký hàm (signature) mới cho callback NetDevice
    bool PacketDropCallback(
        Ptr<NetDevice> device, 
        Ptr<const Packet> packet, 
        uint16_t protocol, 
        const Address& from
    );
};

// =================================================================
// === CÁC HÀM TRIỂN KHAI ===
// =================================================================

DDoSSimulator::DDoSSimulator(uint32_t nIotNodes, uint32_t nAttackers, double simTime)
{
    m_nIotNodes = nIotNodes;
    m_nAttackers = nAttackers;
    m_simTime = simTime;
}

void DDoSSimulator::CreateNodes()
{
    m_iotNodes.Create(m_nIotNodes);
    m_baseStations.Create(2);
    m_serverNode.Create(1);

    m_allNodes.Add(m_iotNodes);
    m_allNodes.Add(m_baseStations);
    m_allNodes.Add(m_serverNode);

    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    for (uint32_t i = 0; i < m_nAttackers; ++i)
    {
        uint32_t attackerIndex = uv->GetInteger(0, m_nIotNodes - 1);
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), attackerIndex) == m_attackerIndices.end())
        {
            m_attackerIndices.push_back(attackerIndex);
        }
        else
        {
            i--; // Thử lại
        }
    }
    NS_LOG_INFO("Created " << m_nIotNodes << " IoT nodes, " << m_attackerIndices.size()
                           << " attackers, 2 base stations, 1 server");
}

void DDoSSimulator::SetupMobility()
{
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(10.0), "MinY", DoubleValue(10.0),
                                  "DeltaX", DoubleValue(15.0), "DeltaY", DoubleValue(15.0),
                                  "GridWidth", UintegerValue(5), "LayoutType", StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                              "Bounds", RectangleValue(Rectangle(0, 100, 0, 100)),
                              "Distance", DoubleValue(5.0));
    mobility.Install(m_iotNodes);

    Ptr<ListPositionAllocator> baseStationPositions = CreateObject<ListPositionAllocator>();
    baseStationPositions->Add(Vector(25.0, 50.0, 0.0));
    baseStationPositions->Add(Vector(75.0, 50.0, 0.0));
    mobility.SetPositionAllocator(baseStationPositions);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(m_baseStations);

    Ptr<ListPositionAllocator> serverPosition = CreateObject<ListPositionAllocator>();
    serverPosition->Add(Vector(50.0, 50.0, 0.0));
    mobility.SetPositionAllocator(serverPosition);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(m_serverNode);
}

void DDoSSimulator::SetupNetwork()
{
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    WifiMacHelper wifiMac;
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());

    wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(Ssid("ddos-network")));
    // Gán vào biến thành viên m_apDevices
    m_apDevices = wifi.Install(wifiPhy, wifiMac, m_baseStations);

    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("ddos-network")));
    NetDeviceContainer staDevices = wifi.Install(wifiPhy, wifiMac, m_iotNodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));

    NetDeviceContainer p2pDevices1 = p2p.Install(m_baseStations.Get(0), m_serverNode.Get(0));
    NetDeviceContainer p2pDevices2 = p2p.Install(m_baseStations.Get(1), m_serverNode.Get(0));

    InternetStackHelper stack;
    stack.Install(m_allNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    m_staInterfaces = address.Assign(staDevices); 
    address.Assign(m_apDevices); // Gán IP cho APs

    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer p2pInterfaces1 = address.Assign(p2pDevices1);
    m_serverIp = p2pInterfaces1.GetAddress(1); 

    address.SetBase("10.1.3.0", "255.255.255.0");
    address.Assign(p2pDevices2);

    NS_LOG_INFO("Populating routing tables...");
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

void DDoSSimulator::SetupApplications()
{
    NS_LOG_INFO("Setting up normal traffic applications...");
    uint16_t serverPort = 8080;
    PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory",
                                      InetSocketAddress(Ipv4Address::GetAny(), serverPort));
    ApplicationContainer sinkApp = packetSinkHelper.Install(m_serverNode);
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(m_simTime));

    for (uint32_t i = 0; i < m_nIotNodes; ++i)
    {
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) !=
            m_attackerIndices.end())
        {
            continue; 
        }
        OnOffHelper onOffHelper("ns3::UdpSocketFactory",
                                InetSocketAddress(m_serverIp, serverPort));
        onOffHelper.SetAttribute("DataRate", StringValue("50kbps"));
        onOffHelper.SetAttribute("PacketSize", UintegerValue(128));
        onOffHelper.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=2.0]"));
        onOffHelper.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=1.0]"));
        ApplicationContainer clientApp = onOffHelper.Install(m_iotNodes.Get(i));
        clientApp.Start(Seconds(1.0 + i * 0.5));
        clientApp.Stop(Seconds(m_simTime - 1.0));
    }
}

void DDoSSimulator::SetupDDoSAttack()
{
    NS_LOG_INFO("Setting up DDoS attack from " << m_attackerIndices.size() << " nodes...");
    uint16_t attackPort = 8080;
    for (uint32_t attackerIndex : m_attackerIndices)
    {
        OnOffHelper attackHelper("ns3::UdpSocketFactory",
                                 InetSocketAddress(m_serverIp, attackPort));
        attackHelper.SetAttribute("DataRate", StringValue("5000kbps"));
        attackHelper.SetAttribute("PacketSize", UintegerValue(1024));
        attackHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=100]"));
        attackHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        ApplicationContainer attackApp = attackHelper.Install(m_iotNodes.Get(attackerIndex));
        attackApp.Start(Seconds(5.0));
        attackApp.Stop(Seconds(m_simTime - 5.0));

        Ipv4Address attackerIp = m_staInterfaces.GetAddress(attackerIndex);
        m_attackerIps.push_back(attackerIp);
        NS_LOG_INFO("Node " << attackerIndex << " (" << attackerIp << ") configured as DDoS attacker");
    }
}

void DDoSSimulator::SetupNetAnim()
{
    NS_LOG_INFO("Setting up NetAnim visualization...");
    static AnimationInterface anim("ddos-animation.xml");
    for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i)
    {
        std::string nodeDescription = "IoT Node ";
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end())
        {
            nodeDescription += "(ATTACKER)";
            anim.UpdateNodeColor(m_iotNodes.Get(i), 255, 0, 0); // Red
        }
        else
        {
            nodeDescription += "(Normal)";
            anim.UpdateNodeColor(m_iotNodes.Get(i), 0, 255, 0); // Green
        }
        anim.UpdateNodeDescription(m_iotNodes.Get(i), nodeDescription);
    }
    for (uint32_t i = 0; i < m_baseStations.GetN(); ++i)
    {
        anim.UpdateNodeColor(m_baseStations.Get(i), 0, 0, 255); // Blue
        anim.UpdateNodeDescription(m_baseStations.Get(i), "Base Station");
    }
    anim.UpdateNodeColor(m_serverNode.Get(0), 128, 0, 128); // Purple
    anim.UpdateNodeDescription(m_serverNode.Get(0), "Server");
}

void DDoSSimulator::CollectStatistics()
{
    NS_LOG_INFO("Collecting simulation statistics...");
    m_monitor = m_flowMonHelper.InstallAll(); 
    m_statsFile.open("realtime_stats.csv");
    m_statsFile << "time,normal_packets,attack_packets,total_throughput,avg_delay\n";

    for (double t = 0.0; t <= m_simTime; t += 1.0)
    {
        Simulator::Schedule(Seconds(t),
                            [this, t]() 
                            {
                                Ptr<Ipv4FlowClassifier> classifier =
                                    DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
                                if (!classifier) return;

                                FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
                                uint32_t currentNormalPackets = 0;
                                uint32_t currentAttackPackets = 0;
                                double totalThroughput = 0;
                                double totalDelay = 0;
                                uint32_t flowCount = 0;

                                for (auto it = stats.begin(); it != stats.end(); ++it)
                                {
                                    if (it->second.rxPackets == 0) continue;
                                    double flowThroughput = it->second.rxBytes * 8.0 / 1000.0;
                                    double flowDelay = it->second.delaySum.GetSeconds();
                                    totalThroughput += flowThroughput;
                                    totalDelay += flowDelay;
                                    flowCount++;
                                    Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
                                    bool isAttackFlow = false;
                                    for (const auto& attackerIp : m_attackerIps)
                                    {
                                        if (t.sourceAddress == attackerIp)
                                        {
                                            isAttackFlow = true;
                                            break;
                                        }
                                    }
                                    uint32_t packetsNow = it->second.txPackets;
                                    uint32_t packetsLast = m_lastTxPackets[it->first]; 
                                    uint32_t deltaPackets = (packetsNow > packetsLast) ? (packetsNow - packetsLast) : 0;
                                    if (isAttackFlow) currentAttackPackets += deltaPackets;
                                    else currentNormalPackets += deltaPackets;
                                    m_lastTxPackets[it->first] = packetsNow;
                                }
                                double avgDelay = (flowCount > 0) ? totalDelay / flowCount : 0;
                                m_statsFile << t << "," << currentNormalPackets << ","
                                            << currentAttackPackets << "," << totalThroughput << ","
                                            << avgDelay << "\n";
                            });
    }

    Simulator::Stop(Seconds(m_simTime));
    Simulator::Run();

    m_statsFile.close();

    m_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    NS_ASSERT_MSG(classifier, "DynamicCast to Ipv4FlowClassifier failed!");

    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
    std::ofstream resultsFile;
    resultsFile.open("ns3_detailed_results.csv");
    resultsFile << "flow_id,source_ip,destination_ip,protocol,tx_packets,rx_packets,"
                << "tx_bytes,rx_bytes,delay_sum,jitter_sum,lost_packets,packet_loss_ratio,"
                << "throughput,flow_duration,label\n";

    uint32_t totalPackets = 0;
    uint32_t attackPackets = 0;
    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        double flowDuration =
            it->second.timeLastRxPacket.GetSeconds() - it->second.timeFirstTxPacket.GetSeconds();
        double throughput = (flowDuration > 0) ? (it->second.rxBytes * 8.0) / (flowDuration * 1000.0) : 0;
        double packetLossRatio = ((it->second.txPackets + it->second.lostPackets) > 0) ? 
            (it->second.lostPackets) / (double)(it->second.txPackets + it->second.lostPackets) : 0;

        bool isAttack = false;
        for (const auto& attackerIp : m_attackerIps)
        {
            if (t.sourceAddress == attackerIp)
            {
                isAttack = true;
                break;
            }
        }
        int label = isAttack ? 1 : 0;
        if (isAttack) attackPackets += it->second.txPackets;
        totalPackets += it->second.txPackets;

        resultsFile << it->first << "," << t.sourceAddress << "," << t.destinationAddress << ","
                    << (int)t.protocol << "," << it->second.txPackets << ","
                    << it->second.rxPackets << "," << it->second.txBytes << ","
                    << it->second.rxBytes << "," << it->second.delaySum.GetSeconds() << ","
                    << it->second.jitterSum.GetSeconds() << "," << it->second.lostPackets << ","
                    << packetLossRatio << "," << throughput << "," << flowDuration << "," << label
                    << "\n";
    }
    resultsFile.close();

    NS_LOG_INFO("Simulation completed:");
    NS_LOG_INFO("  Total packets: " << totalPackets);
    NS_LOG_INFO("  Attack packets: " << attackPackets);
    if (totalPackets > 0)
        NS_LOG_INFO("  Attack percentage: " << (attackPackets * 100.0 / totalPackets) << "%");
    else
        NS_LOG_INFO("  No packets transmitted.");

    NS_LOG_INFO("Results written to ns3_detailed_results.csv");
    NS_LOG_INFO("Real-time stats written to realtime_stats.csv");
    NS_LOG_INFO("NetAnim animation: ddos-animation.xml");

    // <<< VÔ HIỆU HÓA GNUPLOT >>>
    // ExportToGnuplot(); 
    
    Simulator::Destroy();
}

// <<< VÔ HIỆU HÓA TOÀN BỘ HÀM GNUPLOT >>>
/*
void DDoSSimulator::ExportToGnuplot()
{
    NS_LOG_INFO("Exporting data to Gnuplot format...");
    std::ofstream gpFile;
    gpFile.open("ddos_data.dat");
    std::ifstream statsFile("realtime_stats.csv"); 
    std::string line;
    std::getline(statsFile, line); // Bỏ qua header
    gpFile << "# Time NormalPackets AttackPackets Throughput AvgDelay\n";
    while (std::getline(statsFile, line))
    {
        gpFile << line << "\n";
    }
    gpFile.close();
    statsFile.close();
    NS_LOG_INFO("Gnuplot data written to ddos_data.dat");
}
*/

// <<< HÀM RUN() ĐÃ CẬP NHẬT >>>
void DDoSSimulator::Run()
{
    CreateNodes();
    SetupMobility();
    SetupNetwork();
    SetupApplications();
    SetupDDoSAttack();
    SetupNetAnim();
    SetupMitigation(); 
    CollectStatistics();
}

// =================================================================
// <<< CÁC HÀM MỚI CHO VIỆC GIẢM THIỂU (MITIGATION) - ĐÃ SỬA LỖI >>>
// =================================================================

void DDoSSimulator::SetupMitigation()
{
    NS_LOG_INFO("Setting up Mitigation (NetDevice Callback)...");

    // Sử dụng API SetReceiveCallback trên các AP NetDevices
    for (uint32_t i = 0; i < m_apDevices.GetN(); ++i)
    {
        m_apDevices.Get(i)->SetReceiveCallback(
            MakeCallback(&DDoSSimulator::PacketDropCallback, this)
        );
    }

    m_liveStatsFile.open("live_flow_stats.csv", std::ofstream::out | std::ofstream::trunc);
    m_liveStatsFile 
        << "time,source_ip,protocol,tx_packets,rx_packets,tx_bytes,rx_bytes,"
        << "delay_sum,jitter_sum,lost_packets,packet_loss_ratio,"
        << "throughput,flow_duration,label\n";
    m_liveStatsFile.close();

    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
    Simulator::Schedule(Seconds(1.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

bool DDoSSimulator::PacketDropCallback(
    Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from)
{
    Ptr<Packet> p_copy = packet->Copy();
    Ipv4Header header;
    
    // 0x0800 là EtherType (mã giao thức) cho IPv4
    if (protocol == 0x0800) 
    {
        if (p_copy->PeekHeader(header))
        {
            Ipv4Address sourceIp = header.GetSource();
            if (m_blockedIps.count(sourceIp))
            {
                NS_LOG_WARN("MITIGATION: Dropped packet from blocked IP: " << sourceIp);
                return false; // Hủy gói (không nhận)
            }
        }
    }
    return true; // Chấp nhận gói (cho phép xử lý)
}

void DDoSSimulator::MonitorLiveFlows()
{
    if (Simulator::Now().GetSeconds() > m_simTime) return; 

    if (!m_monitor)
    {
        NS_LOG_WARN("MonitorLiveFlows: FlowMonitor chưa được cài đặt (chờ 1s)...");
        Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
        return;
    }

    m_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();

    m_liveStatsFile.open("live_flow_stats.csv", std::ofstream::app);

    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        FlowId flowId = it->first;
        FlowMonitor::FlowStats currentStats = it->second;
        FlowMonitor::FlowStats deltaStats;
        deltaStats.txPackets = currentStats.txPackets - m_lastFlowStats[flowId].txPackets;

        if (deltaStats.txPackets > 0)
        {
            deltaStats.rxPackets = currentStats.rxPackets - m_lastFlowStats[flowId].rxPackets;
            deltaStats.txBytes = currentStats.txBytes - m_lastFlowStats[flowId].txBytes;
            deltaStats.rxBytes = currentStats.rxBytes - m_lastFlowStats[flowId].rxBytes;
            deltaStats.delaySum = currentStats.delaySum - m_lastFlowStats[flowId].delaySum;
            deltaStats.jitterSum = currentStats.jitterSum - m_lastFlowStats[flowId].jitterSum;
            deltaStats.lostPackets = currentStats.lostPackets - m_lastFlowStats[flowId].lostPackets;

            double flowDuration = 1.0; 
            double throughput = (deltaStats.rxBytes * 8.0) / (flowDuration * 1000.0); // Kbps
            double packetLossRatio = 0.0;
            if ((deltaStats.txPackets + deltaStats.lostPackets) > 0)
            {
                packetLossRatio = (deltaStats.lostPackets) / (double)(deltaStats.txPackets + deltaStats.lostPackets);
            }
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);
            bool isAttack = false;
            for (const auto& attackerIp : m_attackerIps)
            {
                if (t.sourceAddress == attackerIp) { isAttack = true; break; }
            }
            m_liveStatsFile
                << Simulator::Now().GetSeconds() << ","
                << t.sourceAddress << "," << (int)t.protocol << ","
                << deltaStats.txPackets << "," << deltaStats.rxPackets << ","
                << deltaStats.txBytes << "," << deltaStats.rxBytes << ","
                << deltaStats.delaySum.GetSeconds() << "," << deltaStats.jitterSum.GetSeconds() << ","
                << deltaStats.lostPackets << "," << packetLossRatio << ","
                << throughput << "," << flowDuration << "," << (isAttack ? 1 : 0) << "\n";
        }
        m_lastFlowStats[flowId] = currentStats; 
    }
    m_liveStatsFile.close();
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
}

void DDoSSimulator::CheckForBlacklistUpdates()
{
    if (Simulator::Now().GetSeconds() > m_simTime) return;

    std::ifstream blacklistFile("blacklist.txt");
    if (blacklistFile.is_open())
    {
        std::string ipString;
        while (std::getline(blacklistFile, ipString))
        {
            if (ipString.empty()) continue;
            Ipv4Address ip;
            ip.Set(ipString.c_str()); 
            if (m_blockedIps.insert(ip).second)
            {
                NS_LOG_INFO("MITIGATION: Đã nhận lệnh chặn IP mới: " << ip);
            }
        }
        blacklistFile.close();
    }
    Simulator::Schedule(Seconds(0.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

// =================================================================
// === HÀM MAIN (KHÔNG THAY ĐỔI) ===
// =================================================================

int main(int argc, char* argv[])
{
    uint32_t nIotNodes = 20;
    uint32_t nAttackers = 5;
    double simTime = 60.0;
    bool enableNetAnim = true;

    CommandLine cmd;
    cmd.AddValue("nodes", "Number of IoT nodes", nIotNodes);
    cmd.AddValue("attackers", "Number of attacker nodes", nAttackers);
    cmd.AddValue("time", "Simulation time in seconds", simTime);
    cmd.AddValue("netanim", "Enable NetAnim visualization", enableNetAnim);
    cmd.Parse(argc, argv);

    LogComponentEnable("DDoSSimulator", LOG_LEVEL_INFO);

    DDoSSimulator simulator(nIotNodes, nAttackers, simTime);
    simulator.Run();

    return 0;
}