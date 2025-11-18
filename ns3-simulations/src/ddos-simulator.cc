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
#include <fstream>
#include <vector>
#include <sstream>
#include <map> 
#include <set> 
#include <iostream> 

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
    NetDeviceContainer m_apDevices; 

    std::vector<uint32_t> m_attackerIndices;
    std::vector<Ipv4Address> m_attackerIps;

    FlowMonitorHelper m_flowMonHelper;
    Ptr<FlowMonitor> m_monitor;

    // Biến thống kê
    std::ofstream m_statsFile; 
    std::map<FlowId, uint32_t> m_lastTxPackets_realtime; 
    std::ofstream m_liveStatsFile; 
    std::set<Ipv4Address> m_blockedIps; 
    std::map<FlowId, uint32_t> m_lastTxPackets_live;

    // --- Các hàm Setup ---
    void CreateNodes();
    void SetupMobility();
    void SetupNetwork();
    void SetupApplications();
    void SetupDDoSAttack();
    void SetupNetAnim();
    
    // <<< HÀM MỚI: Cài đặt FlowMonitor riêng biệt >>>
    void SetupFlowMonitor();

    // --- Các hàm Runtime (Chạy trong lúc mô phỏng) ---
    void ScheduleRealtimeStats();     // Lên lịch ghi realtime_stats.csv
    void SetupMitigation();           // Lên lịch mitigation
    void MonitorLiveFlows();          // Callback cho AI
    void CheckForBlacklistUpdates();  // Callback đọc blacklist
    void UpdateRealtimeStats();       // Callback ghi realtime_stats.csv

    // --- Các hàm Finalize (Sau khi mô phỏng) ---
    void SaveFinalResults();

    bool PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from);
};

// =================================================================
// === IMPLEMENTATION ===
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
    for (uint32_t i = 0; i < m_nAttackers; ++i) {
        uint32_t attackerIndex = uv->GetInteger(0, m_nIotNodes - 1);
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), attackerIndex) == m_attackerIndices.end()) {
            m_attackerIndices.push_back(attackerIndex);
        } else {
            i--; 
        }
    }
    NS_LOG_INFO("Created " << m_nIotNodes << " IoT nodes, " << m_attackerIndices.size() << " attackers.");
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

    Ptr<ListPositionAllocator> staticPositions = CreateObject<ListPositionAllocator>();
    staticPositions->Add(Vector(25.0, 50.0, 0.0)); // BS1
    staticPositions->Add(Vector(75.0, 50.0, 0.0)); // BS2
    staticPositions->Add(Vector(50.0, 50.0, 0.0)); // Server
    
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
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
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

void DDoSSimulator::SetupNetAnim()
{
    static AnimationInterface anim("ddos-animation.xml");
    // (Code NetAnim giữ nguyên, rút gọn để tập trung vào logic chính)
    for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
        anim.UpdateNodeColor(m_iotNodes.Get(i), 0, 255, 0); 
    }
    for (auto idx : m_attackerIndices) {
        anim.UpdateNodeColor(m_iotNodes.Get(idx), 255, 0, 0);
    }
}

// =================================================================
// === CẤU TRÚC MỚI: TÁCH BIỆT FLOW MONITOR VÀ SCHEDULING ===
// =================================================================

// 1. Cài đặt FlowMonitor SỚM
void DDoSSimulator::SetupFlowMonitor()
{
    NS_LOG_INFO("Installing FlowMonitor...");
    m_monitor = m_flowMonHelper.InstallAll();

    // Khởi tạo lastTxPackets cho tất cả flow sau 0.1s
    Simulator::Schedule(Seconds(0.1), [this]() {
        Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
        FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
        for (auto it = stats.begin(); it != stats.end(); ++it)
        {
            m_lastTxPackets_realtime[it->first] = it->second.txPackets;
            m_lastTxPackets_live[it->first] = it->second.txPackets;
        }
    });
}

// 2. Lên lịch ghi Realtime Stats (được gọi riêng, không gộp vào Run)
void DDoSSimulator::ScheduleRealtimeStats()
{
    m_statsFile.open("/home/traphan/ns-3-dev/ddos-project-new/data/raw/realtime_stats.csv",
                     std::ofstream::out | std::ofstream::trunc);
    m_statsFile << "time,normal_packets,attack_packets,total_throughput,avg_delay\n";
    m_statsFile.flush();

    // Gọi lần đầu, callback sẽ tự lặp lại
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::UpdateRealtimeStats, this);
}

// 3. Hàm Callback cập nhật Realtime Stats (Tách riêng)
void DDoSSimulator::UpdateRealtimeStats()
{
    if (!m_monitor) return;

    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();

    uint32_t normalPackets = 0, attackPackets = 0;
    double throughputKbps = 0, totalDelay = 0;
    uint32_t flowCount = 0;

    for (auto it : stats) {
        FlowId flowId = it.first;
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);

        bool isAttack = std::find(m_attackerIps.begin(), m_attackerIps.end(), t.sourceAddress) != m_attackerIps.end();

        // delta packets và delta bytes so với lần trước
        uint32_t deltaPackets = it.second.txPackets - m_lastTxPackets_realtime[flowId];
        uint32_t deltaBytes   = it.second.rxBytes - (m_lastTxPackets_realtime[flowId] * 128); // nếu packet size 128B
        deltaBytes = (deltaBytes > 0) ? deltaBytes : 0;

        if (isAttack) attackPackets += deltaPackets;
        else normalPackets += deltaPackets;

        throughputKbps += deltaBytes * 8.0 / 1000.0; // Kbps
        totalDelay += it.second.delaySum.GetSeconds();
        flowCount++;

        m_lastTxPackets_realtime[flowId] = it.second.txPackets;
    }

    double avgDelay = (flowCount > 0) ? totalDelay / flowCount : 0;
    double now = Simulator::Now().GetSeconds();

    m_statsFile << now << "," << normalPackets << "," << attackPackets << "," 
                << throughputKbps << "," << avgDelay << "\n";
    m_statsFile.flush();

    // Cập nhật NetAnim theo thời gian
    static AnimationInterface anim("ddos-animation.xml");
    for (uint32_t i = 0; i < m_iotNodes.GetN(); ++i) {
        if (std::find(m_attackerIndices.begin(), m_attackerIndices.end(), i) != m_attackerIndices.end())
            anim.UpdateNodeColor(m_iotNodes.Get(i), 255, 0, 0); // attacker
        else
            anim.UpdateNodeColor(m_iotNodes.Get(i), 0, 255, 0); // normal
    }

    // Schedule lần tiếp theo sau 1 giây
    if (now < m_simTime) {
        Simulator::Schedule(Seconds(1.0), &DDoSSimulator::UpdateRealtimeStats, this);
    }
}

// =================================================================
// === MITIGATION LOGIC ===
// =================================================================

void DDoSSimulator::SetupMitigation()
{
    std::cout << "MITIGATION (std::cout): Setting up Mitigation..." << std::endl;

    for (uint32_t i = 0; i < m_allNodes.GetN(); ++i) {
        for (uint32_t j = 0; j < m_allNodes.Get(i)->GetNDevices(); ++j) {
            m_allNodes.Get(i)->GetDevice(j)->SetReceiveCallback(
                MakeCallback(&DDoSSimulator::PacketDropCallback, this)
            );
        }
    }

    m_liveStatsFile.open("/home/traphan/ns-3-dev/ddos-project-new/data/live/live_flow_stats.csv", std::ofstream::out | std::ofstream::trunc);
    m_liveStatsFile << "time,source_ip,protocol,tx_packets,rx_packets,tx_bytes,rx_bytes,"
                    << "delay_sum,jitter_sum,lost_packets,packet_loss_ratio,"
                    << "throughput,flow_duration,label\n";
    m_liveStatsFile.close();

    // callback lặp lại cho live flow
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
    Simulator::Schedule(Seconds(1.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

bool DDoSSimulator::PacketDropCallback(Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address& from)
{
    Ptr<Packet> p_copy = packet->Copy();
    Ipv4Header header;
    if (protocol == 0x0800) {
        if (p_copy->PeekHeader(header)) {
            Ipv4Address sourceIp = header.GetSource();
            if (m_blockedIps.count(sourceIp)) {
                std::cout << "MITIGATION (std::cout): Dropped packet from blocked IP: " << sourceIp << std::endl;
                return true; // HỦY
            }
        }
        Ptr<Ipv4L3Protocol> ipv4l3 = device->GetNode()->GetObject<Ipv4L3Protocol>();
        if (ipv4l3) {
            ipv4l3->Receive(device, packet, protocol, from, device->GetAddress(), NetDevice::PACKET_HOST);
            return true; // CHUYỂN TIẾP
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

    m_liveStatsFile.open("/home/traphan/ns-3-dev/ddos-project-new/data/live/live_flow_stats.csv", std::ofstream::out | std::ofstream::app);

    for (auto it = stats.begin(); it != stats.end(); ++it) {
        FlowId flowId = it->first;
        FlowMonitor::FlowStats currentStats = it->second; 

        double flowDuration = currentStats.timeLastRxPacket.GetSeconds() - currentStats.timeFirstTxPacket.GetSeconds();
        double throughput = (flowDuration > 0) ? (currentStats.rxBytes * 8.0) / (flowDuration * 1000.0) : 0;
        double packetLossRatio = ((currentStats.txPackets + currentStats.lostPackets) > 0) ? 
                                 (currentStats.lostPackets) / (double)(currentStats.txPackets + currentStats.lostPackets) : 0;

        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);
        bool isAttack = std::find(m_attackerIps.begin(), m_attackerIps.end(), t.sourceAddress) != m_attackerIps.end();

        m_liveStatsFile << Simulator::Now().GetSeconds() << ","
                        << t.sourceAddress << "," << (int)t.protocol << ","
                        << currentStats.txPackets << "," << currentStats.rxPackets << ","
                        << currentStats.txBytes << "," << currentStats.rxBytes << ","
                        << currentStats.delaySum.GetSeconds() << "," << currentStats.jitterSum.GetSeconds() << ","
                        << currentStats.lostPackets << "," << packetLossRatio << ","
                        << throughput << "," << flowDuration << "," << (isAttack ? 1 : 0) << "\n";

        m_lastTxPackets_live[flowId] = currentStats.txPackets;
    }

    m_liveStatsFile.close();
    Simulator::Schedule(Seconds(1.0), &DDoSSimulator::MonitorLiveFlows, this);
}

void DDoSSimulator::CheckForBlacklistUpdates()
{
    if (Simulator::Now().GetSeconds() > m_simTime) return;
    std::ifstream blacklistFile("/home/traphan/ns-3-dev/ddos-project-new/data/live/blacklist.txt");
    if (blacklistFile.is_open()) {
        std::string ipString;
        while (std::getline(blacklistFile, ipString)) {
            if (ipString.empty()) continue;
            Ipv4Address ip;
            ip.Set(ipString.c_str()); 
            if (m_blockedIps.insert(ip).second) {
                std::cout << "MITIGATION (std::cout): Đã nhận lệnh chặn IP mới: " << ip << std::endl;
            }
        }
        blacklistFile.close();
    }
    Simulator::Schedule(Seconds(0.5), &DDoSSimulator::CheckForBlacklistUpdates, this);
}

// =================================================================
// === FINAL SAVE (Kết thúc mô phỏng) ===
// =================================================================
void DDoSSimulator::SaveFinalResults()
{
    m_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowMonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = m_monitor->GetFlowStats();
    std::ofstream resultsFile;

    resultsFile.open("/home/traphan/ns-3-dev/ddos-project-new/data/raw/ns3_detailed_results.csv", std::ofstream::out | std::ofstream::trunc); 
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
    NS_LOG_INFO("Simulation completed. Results saved to data/raw/");
}

// =================================================================
// === HÀM RUN MỚI (ĐÚNG LOGIC) ===
// =================================================================
void DDoSSimulator::Run()
{
    CreateNodes();
    SetupMobility();
    SetupNetwork();
    SetupApplications();
    SetupDDoSAttack();
    SetupNetAnim();
    
    // 1. CÀI ĐẶT MONITOR SỚM NHẤT CÓ THỂ
    SetupFlowMonitor();

    // 2. LÊN LỊCH CÁC TÁC VỤ (Nhưng chưa chạy Simulator::Run)
    SetupMitigation(); 
    ScheduleRealtimeStats();

    // 3. CHẠY MÔ PHỎNG
    Simulator::Stop(Seconds(m_simTime));
    Simulator::Run();

    // 4. LƯU KẾT QUẢ SAU CÙNG
    SaveFinalResults();

    Simulator::Destroy();
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