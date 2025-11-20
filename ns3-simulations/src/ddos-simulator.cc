/*
 * DDoS Simulator - FINAL VISUAL UPGRADE
 * - Colors: Neon style for better contrast.
 * - Layout: Optimized spacing for wireless rings visibility.
 * - Logic: Full Mitigation & Bottleneck logic preserved.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/arp-l3-protocol.h"
#include <fstream>
#include <vector>
#include <sstream>
#include <map> 
#include <set> 
#include <iostream> 
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DDoSSimulator");

// Đường dẫn gốc
const std::string PROJECT_ROOT = "/home/traphan/ns-3-dev/ddos-project-new";

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
    NetDeviceContainer m_apDevices; 

    std::vector<uint32_t> m_attackerIndices;
    std::vector<Ipv4Address> m_attackerIps;

    FlowMonitorHelper m_flowMonHelper;
    Ptr<FlowMonitor> m_monitor;

    std::ofstream m_statsFile; 
    std::map<FlowId, uint32_t> m_lastTxPackets_realtime; 
    std::ofstream m_liveStatsFile; 
    std::set<Ipv4Address> m_blockedIps; 
    std::map<FlowId, uint32_t> m_lastTxPackets_live;

    void CreateNodes();
    void SetupMobility();
    void SetupNetwork();
    void SetupApplications();
    void SetupDDoSAttack();
    
    void SetupFlowMonitor();
    void ScheduleRealtimeStats();
    void SetupMitigation();

    void MonitorLiveFlows();          
    void CheckForBlacklistUpdates();  
    void UpdateRealtimeStats();       
    void SaveFinalResults();

    bool PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from);
};

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
    while (m_attackerIndices.size() < m_nAttackers) {
        uint32_t idx = uv->GetInteger(0, m_nIotNodes - 1);
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), idx) == m_attackerIndices.end()) {
            m_attackerIndices.push_back(idx);
        }
    }

    std::cout << "\n===================================================" << std::endl;
    std::cout << "🚧 NETWORK TOPOLOGY INITIALIZED (Visual Enhanced)" << std::endl;
    std::cout << "   - IoT Nodes:   " << m_nIotNodes << " (Left Grid)" << std::endl;
    std::cout << "   - Attackers:   " << m_nAttackers << " (Hidden within IoT)" << std::endl;
    std::cout << "   - Gateways:    2 Base Stations" << std::endl;
    std::cout << "   - Victim:      1 Server" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
}

void DDoSSimulator::SetupMobility()
{
    MobilityHelper mobility;
    
    // 1. IoT Nodes: Xếp thoáng hơn để thấy vòng tròn sóng (Delta = 25)
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0),
                                  "MinY", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(25.0), // Khoảng cách rộng hơn
                                  "DeltaY", DoubleValue(25.0),
                                  "GridWidth", UintegerValue(5), 
                                  "LayoutType", StringValue("RowFirst"));
    
    // Rung nhẹ
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                              "Bounds", RectangleValue(Rectangle(-10, 150, -10, 150)),
                              "Distance", DoubleValue(5.0));
    mobility.Install(m_iotNodes);

    // 2. Base Stations & Server: Sắp xếp cân đối
    Ptr<ListPositionAllocator> staticPositions = CreateObject<ListPositionAllocator>();
    staticPositions->Add(Vector(60.0, 150.0, 0.0));  // BS 1 (Trên)
    staticPositions->Add(Vector(60.0, -30.0, 0.0));  // BS 2 (Dưới)
    staticPositions->Add(Vector(200.0, 60.0, 0.0));  // Server (Xa bên phải)
    
    MobilityHelper staticMobility;
    staticMobility.SetPositionAllocator(staticPositions);
    staticMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    staticMobility.Install(m_baseStations);
    staticMobility.Install(m_serverNode);
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
    m_apDevices = wifi.Install(wifiPhy, wifiMac, m_baseStations);

    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("ddos-network")));
    NetDeviceContainer staDevices = wifi.Install(wifiPhy, wifiMac, m_iotNodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps")); 
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));

    NetDeviceContainer p2pDevices1 = p2p.Install(m_baseStations.Get(0), m_serverNode.Get(0));
    NetDeviceContainer p2pDevices2 = p2p.Install(m_baseStations.Get(1), m_serverNode.Get(0));

    InternetStackHelper stack;
    stack.Install(m_allNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    m_staInterfaces = address.Assign(staDevices); 
    address.Assign(m_apDevices); 

    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer p2pInterfaces1 = address.Assign(p2pDevices1);
    m_serverIp = p2pInterfaces1.GetAddress(1); 

    address.SetBase("10.1.3.0", "255.255.255.0");
    address.Assign(p2pDevices2);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

void DDoSSimulator::SetupApplications()
{
    uint16_t serverPort = 8080;
    PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), serverPort));
    ApplicationContainer sinkApp = packetSinkHelper.Install(m_serverNode);
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(m_simTime));

    for (uint32_t i = 0; i < m_nIotNodes; ++i) {
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end()) continue; 
        
        OnOffHelper onOffHelper("ns3::UdpSocketFactory", InetSocketAddress(m_serverIp, serverPort));
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
    std::cout << "⚔️  ATTACK SCENARIO: UDP Flood (50Mbps) -> 5Mbps Link" << std::endl;
    uint16_t attackPort = 8080;
    for (uint32_t attackerIndex : m_attackerIndices) {
        OnOffHelper attackHelper("ns3::UdpSocketFactory", InetSocketAddress(m_serverIp, attackPort));
        attackHelper.SetAttribute("DataRate", StringValue("5000kbps"));
        attackHelper.SetAttribute("PacketSize", UintegerValue(1024));
        attackHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=100]"));
        attackHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        ApplicationContainer attackApp = attackHelper.Install(m_iotNodes.Get(attackerIndex));
        attackApp.Start(Seconds(5.0));
        attackApp.Stop(Seconds(m_simTime - 5.0));

        Ipv4Address attackerIp = m_staInterfaces.GetAddress(attackerIndex);
        m_attackerIps.push_back(attackerIp);
    }
}

// =================================================================
// === STATS & FLOW MONITOR ===
// =================================================================

void DDoSSimulator::SetupFlowMonitor()
{
    m_monitor = m_flowMonHelper.InstallAll();
}

void DDoSSimulator::ScheduleRealtimeStats()
{
    std::string path = PROJECT_ROOT + "/data/raw/realtime_stats.csv";
    m_statsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc); 
    m_statsFile << "time,normal_packets,attack_packets,total_throughput,avg_delay\n";
    m_statsFile.close();

    for (double t = 1.0; t <= m_simTime; t += 1.0) {
        Simulator::Schedule(Seconds(t), &DDoSSimulator::UpdateRealtimeStats, this);
    }
}

void DDoSSimulator::UpdateRealtimeStats()
{
    if (!m_monitor) return;

    m_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
    
    uint32_t currentNormalPackets = 0;
    uint32_t currentAttackPackets = 0;
    double totalThroughput = 0;
    double totalDelay = 0;
    uint32_t flowCount = 0;

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        if (it->second.txPackets == 0) continue;
        
        double flowThroughput = it->second.rxBytes * 8.0 / 1000.0; 
        double flowDelay = it->second.delaySum.GetSeconds();
        totalThroughput += flowThroughput;
        totalDelay += flowDelay;
        flowCount++;
        
        Ipv4FlowClassifier::FiveTuple t_flow = classifier->FindFlow(it->first);
        bool isAttackFlow = false;
        for (const auto& attackerIp : m_attackerIps) {
            if (t_flow.sourceAddress == attackerIp) { isAttackFlow = true; break; }
        }
        
        uint32_t packetsNow = it->second.txPackets;
        uint32_t packetsLast = m_lastTxPackets_realtime[it->first];
        uint32_t deltaPackets = (packetsNow > packetsLast) ? (packetsNow - packetsLast) : 0;
        
        if (isAttackFlow) currentAttackPackets += deltaPackets;
        else currentNormalPackets += deltaPackets;
        
        m_lastTxPackets_realtime[it->first] = packetsNow;
    }
    
    double avgDelay = (flowCount > 0) ? totalDelay / flowCount : 0;
    double now = Simulator::Now().GetSeconds();

    if (currentAttackPackets > 0 || currentNormalPackets > 0) {
        std::cout << "STATS: Time=" << std::setw(4) << now 
                  << " | Norm=" << std::setw(5) << currentNormalPackets 
                  << " | Attk=" << std::setw(5) << currentAttackPackets 
                  << " | T-put=" << std::setw(8) << totalThroughput << " Kbps" << std::endl;
    }

    std::string path = PROJECT_ROOT + "/data/raw/realtime_stats.csv";
    m_statsFile.open(path.c_str(), std::ofstream::app);
    m_statsFile << now << "," << currentNormalPackets << ","
                << currentAttackPackets << "," << totalThroughput << ","
                << avgDelay << "\n";
    m_statsFile.close();
}

// =================================================================
// === MITIGATION LOGIC ===
// =================================================================

void DDoSSimulator::SetupMitigation()
{
    std::cout << "🛡️  MITIGATION SYSTEM: Initialized" << std::endl;

    for (uint32_t i = 0; i < m_apDevices.GetN(); ++i) {
        m_apDevices.Get(i)->SetReceiveCallback(MakeCallback(&DDoSSimulator::PacketDropCallback, this));
    }

    std::string path = PROJECT_ROOT + "/data/live/live_flow_stats.csv";
    m_liveStatsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    m_liveStatsFile << "time,source_ip,protocol,tx_packets,rx_packets,tx_bytes,rx_bytes,"
                    << "delay_sum,jitter_sum,lost_packets,packet_loss_ratio,"
                    << "throughput,flow_duration,label\n";
    m_liveStatsFile.close();

    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
    Simulator::Schedule(Seconds(1.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

bool DDoSSimulator::PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from)
{
    if (protocol == 0x0800) { 
        Ptr<Packet> p_copy = packet->Copy();
        Ipv4Header header;
        if (p_copy->PeekHeader(header)) {
            Ipv4Address sourceIp = header.GetSource();
            if (m_blockedIps.count(sourceIp)) {
                return true; // DROP
            }
        }
        Ptr<Ipv4L3Protocol> ipv4l3 = device->GetNode()->GetObject<Ipv4L3Protocol>();
        if (ipv4l3) {
            ipv4l3->Receive(device, packet, protocol, from, device->GetAddress(), NetDevice::PACKET_HOST);
            return true; 
        }
    } else if (protocol == 0x0806) { 
        Ptr<ArpL3Protocol> arp = device->GetNode()->GetObject<ArpL3Protocol>();
        if (arp) {
            arp->Receive(device, packet, protocol, from, device->GetAddress(), NetDevice::PACKET_HOST);
            return true; 
        }
    }
    return false; 
}

void DDoSSimulator::MonitorLiveFlows()
{
    if (Simulator::Now().GetSeconds() > m_simTime) return; 
    if (!m_monitor) {
        Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
        return;
    }
    m_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();

    std::string path = PROJECT_ROOT + "/data/live/live_flow_stats.csv";
    m_liveStatsFile.open(path.c_str(), std::ofstream::app);

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        FlowId flowId = it->first;
        FlowMonitor::FlowStats currentStats = it->second; 
        uint32_t packetsNow = currentStats.txPackets;
        uint32_t packetsLast = m_lastTxPackets_live[flowId]; 
        
        if (packetsNow > packetsLast) {
            double flowDuration = currentStats.timeLastRxPacket.GetSeconds() - currentStats.timeFirstTxPacket.GetSeconds();
            double throughput = (flowDuration > 0) ? (currentStats.rxBytes * 8.0) / (flowDuration * 1000.0) : 0;
            double packetLossRatio = 0.0;
            if ((currentStats.txPackets + currentStats.lostPackets) > 0) {
                packetLossRatio = (currentStats.lostPackets) / (double)(currentStats.txPackets + currentStats.lostPackets);
            }
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);
            bool isAttack = false;
            for (const auto& attackerIp : m_attackerIps) {
                if (t.sourceAddress == attackerIp) { isAttack = true; break; }
            }

            m_liveStatsFile << Simulator::Now().GetSeconds() << ","
                            << t.sourceAddress << "," << (int)t.protocol << ","
                            << currentStats.txPackets << "," << currentStats.rxPackets << ","
                            << currentStats.txBytes << "," << currentStats.rxBytes << ","
                            << currentStats.delaySum.GetSeconds() << "," << currentStats.jitterSum.GetSeconds() << ","
                            << currentStats.lostPackets << "," << packetLossRatio << ","
                            << throughput << "," << flowDuration << "," << (isAttack ? 1 : 0) << "\n";
            
            m_lastTxPackets_live[flowId] = packetsNow;
        }
    }
    m_liveStatsFile.close();
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
}

void DDoSSimulator::CheckForBlacklistUpdates()
{
    if (Simulator::Now().GetSeconds() > m_simTime) return;
    std::string path = PROJECT_ROOT + "/data/live/blacklist.txt";
    std::ifstream blacklistFile(path.c_str());
    if (blacklistFile.is_open()) {
        std::string ipString;
        while (std::getline(blacklistFile, ipString)) {
            if (ipString.empty()) continue;
            Ipv4Address ip;
            ip.Set(ipString.c_str()); 
            if (m_blockedIps.insert(ip).second) {
                std::cout << "🚫 BLOCKING: New attacker IP added to blacklist: " << ip << std::endl;
            }
        }
        blacklistFile.close();
    }
    Simulator::Schedule(Seconds(0.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

void DDoSSimulator::SaveFinalResults()
{
    m_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
    std::ofstream resultsFile;

    std::string path = PROJECT_ROOT + "/data/raw/ns3_detailed_results.csv";
    resultsFile.open(path.c_str(), std::ofstream::out | std::ofstream::trunc); 
    resultsFile << "flow_id,source_ip,destination_ip,protocol,tx_packets,rx_packets,"
                << "tx_bytes,rx_bytes,delay_sum,jitter_sum,lost_packets,packet_loss_ratio,"
                << "throughput,flow_duration,label\n";

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        double flowDuration = it->second.timeLastRxPacket.GetSeconds() - it->second.timeFirstTxPacket.GetSeconds();
        double throughput = (flowDuration > 0) ? (it->second.rxBytes * 8.0) / (flowDuration * 1000.0) : 0;
        double packetLossRatio = ((it->second.txPackets + it->second.lostPackets) > 0) ? 
            (it->second.lostPackets) / (double)(it->second.txPackets + it->second.lostPackets) : 0;

        bool isAttack = false;
        for (const auto& attackerIp : m_attackerIps) {
            if (t.sourceAddress == attackerIp) { isAttack = true; break; }
        }
        int label = isAttack ? 1 : 0;

        resultsFile << it->first << "," << t.sourceAddress << "," << t.destinationAddress << ","
                    << (int)t.protocol << "," << it->second.txPackets << ","
                    << it->second.rxPackets << "," << it->second.txBytes << ","
                    << it->second.rxBytes << "," << it->second.delaySum.GetSeconds() << ","
                    << it->second.jitterSum.GetSeconds() << "," << it->second.lostPackets << ","
                    << packetLossRatio << "," << throughput << "," << flowDuration << "," << label
                    << "\n";
    }
    resultsFile.close();
    std::cout << "✅ Simulation completed. Detailed results saved." << std::endl;
}

void DDoSSimulator::Run()
{
    CreateNodes();
    SetupMobility();
    SetupNetwork();
    SetupApplications();
    SetupDDoSAttack();
    
    // --- NETANIM MỚI: ĐẸP & TƯƠNG ĐỐI ---
    AnimationInterface anim("ddos-animation.xml");
    
    // Tắt Metadata để không bị rối và tránh crash khi file quá lớn
    anim.EnablePacketMetadata(false);
    
    // Giới hạn kích thước trace (chỉ ghi 500,000 gói tin đầu)
    anim.SetMaxPktsPerTraceFile(500000);

    // 1. IoT Nodes & Attackers
    for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
        Ptr<Node> n = m_iotNodes.Get(i);
        bool isAttacker = std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end();
        
        if (isAttacker) {
            anim.UpdateNodeColor(n, 255, 0, 0); // ĐỎ
            anim.UpdateNodeDescription(n, "ATTACKER");
            anim.UpdateNodeSize(n, 1.2, 1.2);
        } else {
            anim.UpdateNodeColor(n, 0, 255, 127); // XANH NGỌC
            anim.UpdateNodeDescription(n, "IoT");
            anim.UpdateNodeSize(n, 0.8, 0.8);
        }
    }

    // 2. Base Stations (Nổi bật)
    anim.UpdateNodeColor(m_baseStations.Get(0), 0, 191, 255); // DEEP SKY BLUE
    anim.UpdateNodeSize(m_baseStations.Get(0), 3.0, 3.0);
    anim.UpdateNodeDescription(m_baseStations.Get(0), "Gateway-1");

    anim.UpdateNodeColor(m_baseStations.Get(1), 0, 191, 255); // DEEP SKY BLUE
    anim.UpdateNodeSize(m_baseStations.Get(1), 3.0, 3.0);
    anim.UpdateNodeDescription(m_baseStations.Get(1), "Gateway-2");

    // 3. Server (Rất to và nổi)
    anim.UpdateNodeColor(m_serverNode.Get(0), 255, 215, 0); // VÀNG KIM (GOLD)
    anim.UpdateNodeSize(m_serverNode.Get(0), 5.0, 5.0);
    anim.UpdateNodeDescription(m_serverNode.Get(0), "CLOUD SERVER");

    SetupFlowMonitor();
    ScheduleRealtimeStats();
    SetupMitigation(); 

    std::cout << "\n🚀 STARTING SIMULATION... (Duration: " << m_simTime << "s)" << std::endl;
    Simulator::Stop(Seconds(m_simTime));
    Simulator::Run();
    SaveFinalResults();
    Simulator::Destroy();
    std::cout << "✅ SIMULATION FINISHED. Animation saved to 'ddos-animation.xml'" << std::endl;
}

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